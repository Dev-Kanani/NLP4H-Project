"""Stage 2 - multi-style anatomical sectionizer.

Three strategies are tried in order of reliability and combined per report:

1. **Header matching** (Structured style): canonical organ headers at line
   start - ``LIVER:``, ``GALLBLADDER:``, ``IMPRESSION:``.
2. **Abbreviation matching** (Telegraphic style): shorthand headers - ``GB:``,
   ``PAN:``, ``SPL:``, ``RK:``, ``LK:``, ``R kidney:``, ``Imp:``, and combined
   headers such as ``PAN/SPL:`` whose clauses are then routed individually.
3. **Semantic clause routing** (Unstructured style): when no headers exist the
   narrative is decomposed into sentences, sentences into clauses at
   coordinating boundaries, and each clause is routed to the organ named by its
   first anatomical anchor.  Anchor-less clauses inherit the previous organ
   (radiologists rarely repeat the subject: "...no hydronephrosis is present;
   however, a tiny calculus cannot be excluded").  Summary clauses ("Overall,
   ...") are routed to Impression.

A section may therefore consist of several non-contiguous *segments* of the
original text; segments carry original character offsets so every evidence
span stays traceable to the source report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .schemas import Laterality, Section, SectionName

# --------------------------------------------------------------------------- header table

_HEADER_ALIASES: dict[str, tuple[SectionName, Optional[Laterality], str]] = {}


def _register(names: tuple[str, ...], section: SectionName, method: str, side: Optional[Laterality] = None) -> None:
    for name in names:
        _HEADER_ALIASES[name.lower()] = (section, side, method)


_register(("liver", "hepatic", "hepatobiliary"), SectionName.LIVER, "header")
_register(("gallbladder", "gall bladder", "biliary"), SectionName.GALLBLADDER, "header")
_register(("gb", "g.b.", "g/b"), SectionName.GALLBLADDER, "abbreviation")
_register(("pancreas",), SectionName.PANCREAS, "header")
_register(("pan", "panc"), SectionName.PANCREAS, "abbreviation")
_register(("spleen",), SectionName.SPLEEN, "header")
_register(("spl",), SectionName.SPLEEN, "abbreviation")
_register(("kidneys", "kidney", "renal", "both kidneys", "kidneys and ureters"), SectionName.KIDNEYS, "header")
_register(("kub",), SectionName.KIDNEYS, "abbreviation")
_register(("right kidney", "rt kidney", "rt. kidney"), SectionName.KIDNEYS, "header", Laterality.RIGHT)
_register(("left kidney", "lt kidney", "lt. kidney"), SectionName.KIDNEYS, "header", Laterality.LEFT)
_register(("r kidney", "rk", "r. kidney"), SectionName.KIDNEYS, "abbreviation", Laterality.RIGHT)
_register(("l kidney", "lk", "l. kidney"), SectionName.KIDNEYS, "abbreviation", Laterality.LEFT)
_register(
    ("pelvis", "urinary bladder", "bladder", "prostate", "uterus", "ovaries", "adnexa", "pelvic organs"),
    SectionName.PELVIS, "header",
)
_register(("ub",), SectionName.PELVIS, "abbreviation")
_register(("impression", "conclusion", "opinion", "summary", "final impression"), SectionName.IMPRESSION, "header")
_register(("imp", "impr"), SectionName.IMPRESSION, "abbreviation")

_ALIAS_ALT = "|".join(sorted((re.escape(a) for a in _HEADER_ALIASES), key=len, reverse=True))
# One header = one or more aliases joined by "/" "&" "and" ",", then ":" or " -".
HEADER_RE = re.compile(
    rf"^[ \t]*(?P<header>(?:{_ALIAS_ALT})(?:\s*(?:/|&|,|and)\s*(?:{_ALIAS_ALT}))*)[ \t]*(?::|\s-\s)[ \t]*",
    re.I | re.M,
)

# --------------------------------------------------------------------------- semantic routing

ORGAN_ANCHORS: tuple[tuple[SectionName, re.Pattern[str]], ...] = (
    (SectionName.GALLBLADDER, re.compile(r"gall\s?bladder|\bGB\b|cholecyst\w*|murphy", re.I)),
    (SectionName.LIVER, re.compile(r"\bliver\b|\bhepat\w*|steatosis|fatty\s+(?:liver|infiltration|change)", re.I)),
    (SectionName.PANCREAS, re.compile(r"pancrea\w*", re.I)),
    (SectionName.SPLEEN, re.compile(r"\bspleen\b|\bsplenic\b|splenomegaly", re.I)),
    (SectionName.KIDNEYS, re.compile(r"\bkidneys?\b|\brenal\b|hydronephrosis|nephro\w*", re.I)),
    (SectionName.PELVIS, re.compile(
        r"\bpelvi[sc]\b|urinary\s+bladder|\bbladder\b|prostat\w*|uter\w*|ovar\w*|adnex\w*|ascites|free\s+fluid",
        re.I)),
)
SUMMARY_CUE = re.compile(r"^\s*(?:overall|in\s+summary|in\s+conclusion|impression|conclusion)\b", re.I)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n+")
CLAUSE_SPLIT = re.compile(
    r";\s*|,\s+(?=(?:and|while|whereas|but|although|though|however)\s+(?:the|both|no|a|an|its|this|although|assessment)\b)",
    re.I,
)


@dataclass
class _SectionBuilder:
    name: SectionName
    method: str
    header: Optional[str] = None
    laterality: Optional[Laterality] = None
    segments: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class SectionedReport:
    """Sections plus the original-offset segments that compose each one."""

    text: str
    sections: list[Section]
    segments: dict[int, list[tuple[int, int]]]  # index into ``sections`` -> spans
    style_detected: str

    def iter_segments(self):
        for idx, section in enumerate(self.sections):
            for start, end in self.segments[idx]:
                yield section, start, end


class AnatomicalSectionizer:
    def __init__(self, min_headers_for_header_mode: int = 2) -> None:
        self.min_headers = min_headers_for_header_mode

    # ------------------------------------------------------------------ public API
    def sectionize(self, text: str) -> SectionedReport:
        headers = list(HEADER_RE.finditer(text))
        if len(headers) >= self.min_headers:
            builders = self._from_headers(text, headers)
            methods = {b.method for b in builders}
            style = "Telegraphic" if "abbreviation" in methods else "Structured"
        else:
            builders = self._route_clauses(text, 0, len(text), default=None, method="semantic")
            style = "Unstructured"
        return self._finalise(text, builders, style)

    # ------------------------------------------------------------------ header mode
    @staticmethod
    def _resolve_header(raw: str) -> list[tuple[SectionName, Optional[Laterality], str]]:
        parts = re.split(r"\s*(?:/|&|,|\band\b)\s*", raw.strip(), flags=re.I)
        resolved = [_HEADER_ALIASES[p.lower()] for p in parts if p.lower() in _HEADER_ALIASES]
        if not resolved and raw.lower() in _HEADER_ALIASES:
            resolved = [_HEADER_ALIASES[raw.lower()]]
        return resolved

    def _from_headers(self, text: str, headers: list[re.Match[str]]) -> list[_SectionBuilder]:
        builders: list[_SectionBuilder] = []
        preamble_end = headers[0].start()
        if text[:preamble_end].strip():
            builders.extend(self._route_clauses(text, 0, preamble_end, default=None, method="semantic"))
        for i, match in enumerate(headers):
            body_start = match.end()
            body_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            while body_end > body_start and text[body_end - 1].isspace():
                body_end -= 1
            if body_end <= body_start:
                continue
            raw_header = match.group("header")
            targets = self._resolve_header(raw_header)
            if not targets:
                continue
            if len(targets) == 1:
                name, side, method = targets[0]
                builders.append(_SectionBuilder(name, method, raw_header, side, [(body_start, body_end)]))
            else:
                # Combined header (e.g. "PAN/SPL:"): route clauses among the named organs only.
                allowed = {t[0] for t in targets}
                method = "abbreviation" if any(t[2] == "abbreviation" for t in targets) else "header"
                routed = self._route_clauses(
                    text, body_start, body_end, default=targets[0][0], method=method, allowed=allowed
                )
                for b in routed:
                    b.header = raw_header
                builders.extend(routed)
        return builders

    # ------------------------------------------------------------------ semantic mode
    @staticmethod
    def _clauses(text: str, start: int, end: int) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        chunk = text[start:end]
        sentences: list[tuple[int, int]] = []
        cursor = 0
        for m in SENTENCE_SPLIT.finditer(chunk):
            sentences.append((cursor, m.start()))
            cursor = m.end()
        sentences.append((cursor, len(chunk)))
        for s_start, s_end in sentences:
            sentence = chunk[s_start:s_end]
            clause_start = 0
            for m in CLAUSE_SPLIT.finditer(sentence):
                spans.append((start + s_start + clause_start, start + s_start + m.start()))
                clause_start = m.end()
            spans.append((start + s_start + clause_start, start + s_end))
        # trim whitespace and drop empties
        trimmed = []
        for s, e in spans:
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if e > s:
                trimmed.append((s, e))
        return trimmed

    @staticmethod
    def _anchors(clause: str, allowed: Optional[set[SectionName]]) -> list[SectionName]:
        found: list[tuple[int, SectionName]] = []
        for name, regex in ORGAN_ANCHORS:
            if allowed is not None and name not in allowed:
                continue
            m = regex.search(clause)
            if m:
                found.append((m.start(), name))
        return [name for _, name in sorted(found)]

    def _route_clauses(
        self,
        text: str,
        start: int,
        end: int,
        default: Optional[SectionName],
        method: str,
        allowed: Optional[set[SectionName]] = None,
    ) -> list[_SectionBuilder]:
        builders: dict[SectionName, _SectionBuilder] = {}
        order: list[SectionName] = []
        current: Optional[SectionName] = default
        for c_start, c_end in self._clauses(text, start, end):
            clause = text[c_start:c_end]
            if allowed is None and SUMMARY_CUE.search(clause):
                targets = [SectionName.IMPRESSION]
                current = SectionName.IMPRESSION
            else:
                anchors = self._anchors(clause, allowed)
                if anchors:
                    targets = anchors if self._is_shared_subject(clause, anchors) else [anchors[0]]
                    current = anchors[0]
                elif current is not None:
                    targets = [current]
                else:
                    continue
            for name in targets:
                if name not in builders:
                    builders[name] = _SectionBuilder(name, method)
                    order.append(name)
                builders[name].segments.append((c_start, c_end))
        return [builders[n] for n in order]

    @staticmethod
    def _is_shared_subject(clause: str, anchors: list[SectionName]) -> bool:
        """'The visualized pancreas and spleen are unremarkable' -> both organs share the predicate."""
        if len(anchors) < 2:
            return False
        return bool(re.search(r"\b(?:pancreas|liver|spleen|gallbladder|kidneys?)\s+and\s+(?:the\s+)?"
                              r"(?:pancreas|liver|spleen|gallbladder|kidneys?)\s+(?:are|appear|show)\b", clause, re.I))

    # ------------------------------------------------------------------ assembly
    @staticmethod
    def _finalise(text: str, builders: list[_SectionBuilder], style: str) -> SectionedReport:
        sections: list[Section] = []
        segments: dict[int, list[tuple[int, int]]] = {}
        for b in builders:
            if not b.segments:
                continue
            seg_text = " ".join(text[s:e] for s, e in b.segments)
            sections.append(
                Section(
                    name=b.name, text=seg_text, start=b.segments[0][0], end=b.segments[-1][1],
                    header=b.header, laterality=b.laterality, method=b.method,  # type: ignore[arg-type]
                )
            )
            segments[len(sections) - 1] = list(b.segments)
        return SectionedReport(text=text, sections=sections, segments=segments, style_detected=style)
