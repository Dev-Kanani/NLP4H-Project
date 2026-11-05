"""Deterministic verification of model-proposed findings.

An LLM is fluent but not accountable, so nothing it proposes reaches the
database unchecked.  Each proposed finding passes these gates:

1. **Vocabulary** - organ and finding names are normalised onto the ontology
   (exact -> alias -> fuzzy -> organ correction).  Unmatched names are rejected
   as ``out_of_vocabulary``.
2. **Grounding** - the quoted evidence must exist in the source report: exact,
   then case/whitespace-insensitive, then fuzzy (token-window similarity >=
   threshold).  Ungrounded findings are rejected as ``ungrounded_evidence``;
   their rate is the hallucination rate.  The stored evidence is always the
   *source* text at the matched offsets, never the model's paraphrase.
3. **Attributes** - grade kept only for non-negated steatosis; size kept only
   for sizeable concepts, non-negated, and only if that number (in mm or cm)
   appears near the evidence.
4. **Assertion cross-check** - the ConText engine re-reads the evidence; a
   disagreement is *flagged* (``assertion_disagreement:<context status>``), not
   overridden, so analysts can audit model negation errors.
5. **Consistency** - exact duplicates (same organ, finding, status and span)
   are dropped; "Unremarkable" next to an abnormality in the same organ is
   flagged.
6. **Grounding to codes / triage** from the ontology, then Pydantic validation.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import ValidationError

from .context import ConTextEngine
from .ontology import Ontology, load_ontology
from .schemas import Assertion, Finding, Grade, Laterality, Organ, TriageTier

_TOKEN = re.compile(r"\S+")
_NUM = re.compile(r"(\d+(?:\.\d+)?)")
_GRADE_WORDS = {"1": "Grade 1", "i": "Grade 1", "mild": "Grade 1", "2": "Grade 2", "ii": "Grade 2",
                "moderate": "Grade 2", "3": "Grade 3", "iii": "Grade 3", "severe": "Grade 3", "marked": "Grade 3"}
_STATUS_WORDS = {"affirmed": "Affirmed", "present": "Affirmed", "positive": "Affirmed", "yes": "Affirmed",
                 "negated": "Negated", "absent": "Negated", "negative": "Negated", "no": "Negated",
                 "possible": "Possible", "uncertain": "Possible", "probable": "Possible", "suspected": "Possible"}


@dataclass
class GroundedSpan:
    start: int
    end: int
    method: str  # exact | normalized | fuzzy
    score: float


@dataclass
class VerificationResult:
    findings: list[Finding]
    rejected: list[dict[str, Any]] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)
    proposed: int = 0


def locate_evidence(text: str, evidence: str, threshold: float = 0.85,
                    avoid: frozenset[int] = frozenset()) -> Optional[GroundedSpan]:
    """Find ``evidence`` in ``text``; returns source offsets or ``None`` if not grounded.

    ``avoid`` holds start offsets already claimed by an identical finding, so a phrase
    repeated on two lines ("RK: No calculus/HDN" ... "LK: No calculus/HDN") grounds to
    its second occurrence instead of collapsing into the first.
    """
    ev = (evidence or "").strip().strip('"').strip()
    if len(ev) < 2:
        return None
    i = text.find(ev)
    first_exact = i
    while i >= 0 and i in avoid:
        i = text.find(ev, i + 1)
    if i >= 0:
        return GroundedSpan(i, i + len(ev), "exact", 1.0)
    if first_exact >= 0:
        return GroundedSpan(first_exact, first_exact + len(ev), "exact", 1.0)
    # case / whitespace-insensitive: map normalised positions back to source offsets
    norm_chars, index = [], []
    prev_space = False
    for pos, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            norm_chars.append(" "); index.append(pos); prev_space = True
        else:
            norm_chars.append(ch.lower()); index.append(pos); prev_space = False
    norm_text = "".join(norm_chars)
    norm_ev = re.sub(r"\s+", " ", ev.lower())
    j = norm_text.find(norm_ev)
    if j >= 0:
        return GroundedSpan(index[j], index[j + len(norm_ev) - 1] + 1, "normalized", 1.0)
    # fuzzy: best token window of similar length
    toks = [(m.start(), m.end()) for m in _TOKEN.finditer(text)]
    n_ev = len(norm_ev.split())
    if not toks or n_ev == 0:
        return None
    best: Optional[GroundedSpan] = None
    for width in range(max(1, n_ev - 2), n_ev + 3):
        for k in range(0, max(1, len(toks) - width + 1)):
            s, e = toks[k][0], toks[min(k + width, len(toks)) - 1][1]
            cand = re.sub(r"\s+", " ", text[s:e].lower())
            ratio = difflib.SequenceMatcher(None, cand, norm_ev, autojunk=False).ratio()
            if ratio >= threshold and (best is None or ratio > best.score):
                best = GroundedSpan(s, e, "fuzzy", ratio)
    return best


class FindingVerifier:
    def __init__(self, ontology: Optional[Ontology] = None, context: Optional[ConTextEngine] = None,
                 fuzzy_threshold: float = 0.85) -> None:
        self.ontology = ontology or load_ontology()
        self.context = context or ConTextEngine()
        self.fuzzy_threshold = fuzzy_threshold

    # ------------------------------------------------------------------ attribute parsing
    @staticmethod
    def _status(raw: Any) -> Optional[str]:
        return _STATUS_WORDS.get(str(raw or "").strip().lower())

    @staticmethod
    def _grade(raw: Any) -> str:
        if raw in (None, "", "N/A", "null"):
            return "N/A"
        s = str(raw).strip().lower().replace("grade", "").replace("gr", "").strip(" -.")
        return _GRADE_WORDS.get(s, "N/A")

    @staticmethod
    def _size(raw: Any) -> Optional[float]:
        if raw in (None, "", "null"):
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        nums = [float(x) for x in _NUM.findall(str(raw))]
        if not nums:
            return None
        value = max(nums)
        return value * 10 if re.search(r"\bcm\b", str(raw), re.I) else value

    @staticmethod
    def _size_in_text(size: float, window: str) -> bool:
        nums = {float(x) for x in _NUM.findall(window)}
        return any(abs(n - size) < 0.051 or abs(n * 10 - size) < 0.51 for n in nums)

    def _context_status(self, text: str, span: GroundedSpan, organ: str, finding: str) -> Optional[str]:
        """Re-assert the finding with ConText inside its evidence sentence, if the term can be located."""
        entry = self.ontology.get(organ, finding)
        if entry is None or entry.concept in {"Unremarkable", "Unremarkable Study"}:
            return None
        s_start = max(text.rfind(".", 0, span.start), text.rfind("\n", 0, span.start)) + 1
        s_end_candidates = [i for i in (text.find(".", span.end), text.find("\n", span.end)) if i >= 0]
        s_end = min(s_end_candidates) if s_end_candidates else len(text)
        sentence = text[s_start:s_end]
        lowered = sentence.lower()
        terms = sorted({finding.lower(), *[a.lower() for a in entry.aliases]}, key=len, reverse=True)
        for term in terms:
            for m in re.finditer(rf"\b{re.escape(term)}\b", lowered):
                abs_start = s_start + m.start()
                if span.start - 5 <= abs_start <= span.end + 5:
                    return self.context.assert_target(sentence, m.start(), m.end())[0].value
        return None

    # ------------------------------------------------------------------ main entry
    def verify(self, report_id: str, text: str, items: list[dict[str, Any]], extractor: str) -> VerificationResult:
        res = VerificationResult(findings=[], proposed=len(items))
        accepted: list[dict[str, Any]] = []
        seen: set[tuple] = set()

        for raw in items:
            def reject(reason: str, **extra) -> None:
                res.rejected.append({"report_id": report_id, "reason": reason, "item": raw, **extra})

            if not isinstance(raw, dict):
                reject("malformed_item"); continue
            organ = self.ontology.normalize_organ(str(raw.get("organ", "")))
            if organ is None:
                reject("unknown_organ"); continue
            finding_raw = str(raw.get("finding", ""))
            if self.ontology.study_level(finding_raw) or organ == "Global":
                organ, finding, method = "Global", ("Unremarkable Study" if organ == "Global" or
                                                   self.ontology.study_level(finding_raw) else None), "study"
                if finding is None:
                    reject("out_of_vocabulary"); continue
            else:
                organ, finding, method = self.ontology.normalize_finding(organ, finding_raw)
            if finding is None:
                reject("out_of_vocabulary"); continue
            status = self._status(raw.get("status"))
            if status is None:
                reject("invalid_status"); continue
            claimed = frozenset(k[3] for k in seen if k[:3] == (organ, finding, status))
            span = locate_evidence(text, str(raw.get("evidence", "")), self.fuzzy_threshold, avoid=claimed)
            if span is None:
                reject("ungrounded_evidence"); continue

            flags: list[str] = []
            if method not in {"exact", "study"}:
                flags.append(f"vocabulary_{method}")
            if span.method == "fuzzy":
                flags.append(f"evidence_fuzzy_{span.score:.2f}")

            entry = self.ontology.get(organ, finding)
            grade = self._grade(raw.get("grade"))
            if grade != "N/A" and (entry.concept != "Steatosis" or status == "Negated"):
                flags.append("grade_dropped"); grade = "N/A"

            size = self._size(raw.get("size_mm"))
            if size is not None:
                window = text[max(0, span.start - 150): span.end + 150]
                if not entry.sizeable or status == "Negated" or not (0 < size <= 300):
                    flags.append("size_dropped"); size = None
                elif not self._size_in_text(size, window):
                    flags.append("size_not_in_text"); size = None

            lat = raw.get("laterality")
            laterality = lat if lat in {"Left", "Right", "Bilateral"} else None

            ctx = self._context_status(text, span, organ, finding)
            if ctx is not None and ctx != status:
                flags.append(f"assertion_disagreement:{ctx}")

            key = (organ, finding, status, span.start, span.end)
            if key in seen:
                reject("duplicate"); continue
            seen.add(key)
            accepted.append({"organ": organ, "finding": finding, "concept": entry.concept, "status": status,
                             "grade": grade, "size_mm": size, "laterality": laterality, "span": span,
                             "flags": flags})

        # consistency: Unremarkable alongside an abnormality in the same organ
        abnormal_organs = {a["organ"] for a in accepted
                           if self.ontology.is_pathological(a["organ"], a["finding"]) and a["status"] != "Negated"}
        for a in accepted:
            if a["concept"] == "Unremarkable" and a["organ"] in abnormal_organs:
                a["flags"].append("unremarkable_with_abnormality")

        order = {o.value: i for i, o in enumerate(Organ)}
        accepted.sort(key=lambda a: (order[a["organ"]], a["span"].start))
        for n, a in enumerate(accepted, start=1):
            radlex, icd = self.ontology.codes(a["organ"], a["finding"], a["status"])
            rule = self.ontology.triage(a["organ"], a["finding"], a["status"], a["grade"])
            try:
                res.findings.append(Finding(
                    report_id=report_id, finding_id=f"{report_id}_P{n:02d}", organ=Organ(a["organ"]),
                    finding=a["finding"], concept=a["concept"], grade=Grade(a["grade"]), size_mm=a["size_mm"],
                    laterality=Laterality(a["laterality"]) if a["laterality"] else None, location=None,
                    status=Assertion(a["status"]), radlex_id=radlex, icd10_code=icd,
                    evidence_text=text[a["span"].start:a["span"].end], span_start=a["span"].start,
                    span_end=a["span"].end, section=a["organ"],
                    triage_tier=TriageTier(rule[0]) if rule else TriageTier.UNREMARKABLE,
                    extractor=extractor, flags=a["flags"],
                ))
            except ValidationError as exc:
                res.schema_errors.append(f"{report_id} {a['organ']}/{a['finding']}: {exc.errors()[0]['msg']}")
        return res
