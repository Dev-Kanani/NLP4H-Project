"""Stage 3 - clinical information extraction, assertion and ontology grounding.

For every organ section the extractor

1. splits the section into sentences (decimal-safe),
2. matches the organ-scoped target gazetteer (longest match wins on overlap),
3. assigns an assertion with the ConText engine,
4. attaches attributes - severity grade, size in mm, laterality, location,
5. grounds the concept to RadLex / ICD-10-CM, and
6. de-duplicates repeated mentions inside the same section.

Two rule families then add normality statements:

* explicit organ-level "unremarkable" (e.g. "Visualized portions are
  unremarkable") -> ``Unremarkable``; when the organ was partly obscured by
  bowel gas the record is qualified as ``Unremarkable Visualized Portion``;
* a study-level summary ("Overall, this is an unremarkable abdominal
  ultrasound study") -> ``Unremarkable Study`` (organ ``Global``), whose
  normality is then inherited by each organ described with positive normal
  morphology and carrying no pathology.

Impression text is used *only* for the study-level statement, never to create
duplicate organ findings - impressions restate the body of the report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .context import ConTextEngine, TriggerHit
from .lexicon import (
    EXPLICIT_UNREMARKABLE,
    GRADE_ADJECTIVES,
    GRADE_EXPLICIT,
    LATERALITY,
    LOCATION,
    MEASUREMENT,
    NORMALITY_DESCRIPTORS,
    ROMAN_TO_GRADE,
    STUDY_LEVEL_BLOCKERS,
    TARGETS,
    UNREMARKABLE_STUDY,
    TargetConcept,
)
from .ontology import load_ontology
from .schemas import Assertion, Finding, Grade, Laterality, Organ, Section, SectionName, TriageTier
from .sectionizer import SENTENCE_SPLIT, SectionedReport

LATERALISED_ORGANS = frozenset({Organ.KIDNEYS, Organ.PELVIS})
_CLAUSE_BREAK = re.compile(r";|\bbut\b|\bhowever\b|\balthough\b|\bdespite\b", re.I)
_SIZE_CONTINUATION = re.compile(r";\s*(?:the\s+)?(?:largest|biggest|measur\w*|size)\b", re.I)


@dataclass
class CandidateFinding:
    """Pre-validation record; becomes a :class:`Finding` after Pydantic checks."""

    organ: Organ
    finding: str
    concept: str
    status: Assertion
    evidence_text: str
    span_start: int
    span_end: int
    section_label: str
    pathological: bool
    grade: Grade = Grade.NA
    size_mm: Optional[float] = None
    laterality: Optional[Laterality] = None
    location: Optional[str] = None
    radlex_id: Optional[str] = None
    icd10_code: Optional[str] = None
    trigger: Optional[str] = None
    extras: dict[str, str] = field(default_factory=dict)


@dataclass
class _TargetHit:
    start: int
    end: int
    concept: TargetConcept


class ClinicalExtractor:
    def __init__(self, context_engine: ConTextEngine | None = None, targets: Iterable[TargetConcept] = TARGETS) -> None:
        self.context = context_engine or ConTextEngine()
        self.targets = tuple(targets)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _organ(section: Section) -> Optional[Organ]:
        return None if section.name is SectionName.IMPRESSION else Organ(section.name.value)

    @staticmethod
    def _sentences(text: str, start: int, end: int) -> list[tuple[int, int]]:
        spans, cursor = [], start
        for m in SENTENCE_SPLIT.finditer(text, start, end):
            spans.append((cursor, m.start()))
            cursor = m.end()
        spans.append((cursor, end))
        out = []
        for s, e in spans:
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1] in " \t\n.":
                e -= 1
            if e > s:
                out.append((s, e))
        return out

    def _match_targets(self, sentence: str, organ: Organ) -> list[_TargetHit]:
        hits: list[_TargetHit] = []
        for concept in self.targets:
            if organ not in concept.organs:
                continue
            for regex in concept.compiled:
                for m in regex.finditer(sentence):
                    s, e = m.span()
                    while s < e and sentence[s].isspace():
                        s += 1
                    if e > s:
                        hits.append(_TargetHit(s, e, concept))
        hits.sort(key=lambda h: (-(h.end - h.start), h.start))
        accepted: list[_TargetHit] = []
        for h in hits:
            if all(h.end <= a.start or h.start >= a.end for a in accepted):
                accepted.append(h)
        return sorted(accepted, key=lambda h: h.start)

    @staticmethod
    def _grade(sentence: str, anchor: int) -> Grade:
        explicit = GRADE_EXPLICIT.search(sentence)
        if explicit:
            return Grade(ROMAN_TO_GRADE[explicit.group("g").upper()])
        best: tuple[int, str] | None = None
        for regex, label in GRADE_ADJECTIVES:
            for m in regex.finditer(sentence):
                dist = abs(m.start() - anchor)
                if best is None or dist < best[0]:
                    best = (dist, label)
        return Grade(best[1]) if best else Grade.NA

    @staticmethod
    def _clause_bounds(sentence: str, start: int, end: int) -> tuple[int, int]:
        left = 0
        for m in _CLAUSE_BREAK.finditer(sentence, 0, start):
            left = m.end()
        right_match = _CLAUSE_BREAK.search(sentence, end)
        return left, right_match.start() if right_match else len(sentence)

    def _size_mm(self, sentence: str, start: int, end: int) -> Optional[float]:
        c_left, c_right = self._clause_bounds(sentence, start, end)
        # "...calculi are present; the largest measures 12.0 mm" - the size clause follows the semicolon
        if _SIZE_CONTINUATION.match(sentence, c_right):
            c_right = self._clause_bounds(sentence, c_right + 1, c_right + 1)[1]
        best: tuple[int, float] | None = None
        for m in MEASUREMENT.finditer(sentence, c_left, c_right):
            dims = [float(m.group("a"))] + ([float(m.group("b"))] if m.group("b") else [])
            value = max(dims) * (10.0 if m.group("unit").lower() == "cm" else 1.0)
            # prefer measurements *after* the target ("calculus measuring 6.5 mm"), then nearest
            dist = (m.start() - end) if m.start() >= end else (start - m.end()) + 1000
            if best is None or dist < best[0]:
                best = (dist, round(value, 1))
        return best[1] if best else None

    def _location(self, sentence: str, start: int, end: int) -> Optional[str]:
        c_left, c_right = self._clause_bounds(sentence, start, end)
        m = LOCATION.search(sentence, c_left, c_right)
        return re.sub(r"[\s\-]+", " ", m.group("loc").lower()) if m else None

    def _laterality(self, sentence: str, start: int, end: int, section: Section) -> Optional[Laterality]:
        if section.laterality:
            return section.laterality
        # 1) modifier immediately before/inside the target ("tiny left renal calculus")
        found = [m.group("side").lower() for m in LATERALITY.finditer(sentence[max(0, start - 30) : end])]
        side = found[-1] if found else None
        if side is None:
            # 2) prepositional attachment after the target ("cyst ... of the left kidney"), same clause only
            _, c_right = self._clause_bounds(sentence, start, end)
            after = LATERALITY.search(sentence, end, c_right)
            side = after.group("side").lower() if after else None
        if side is None:
            return None
        if side.startswith("bilateral") or side == "both":
            return Laterality.BILATERAL
        return Laterality.LEFT if side == "left" else Laterality.RIGHT

    # ------------------------------------------------------------------ organ sections
    def _extract_section(self, text: str, section: Section, spans: list[tuple[int, int]]) -> list[CandidateFinding]:
        organ = self._organ(section)
        if organ is None:
            return []
        found: list[CandidateFinding] = []
        for seg_start, seg_end in spans:
            for s_start, s_end in self._sentences(text, seg_start, seg_end):
                sentence = text[s_start:s_end]
                triggers = self.context.find_triggers(sentence)
                for hit in self._match_targets(sentence, organ):
                    found.append(self._build(sentence, s_start, hit, triggers, section, organ))
        return self._dedupe(found)

    def _build(
        self, sentence: str, offset: int, hit: _TargetHit, triggers: list[TriggerHit], section: Section, organ: Organ
    ) -> CandidateFinding:
        concept = hit.concept
        status, trigger = self.context.assert_target(sentence, hit.start, hit.end, triggers)
        cand = CandidateFinding(
            organ=organ,
            finding=concept.finding,
            concept=concept.concept,
            status=status,
            evidence_text=sentence.strip(),
            span_start=offset,
            span_end=offset + len(sentence),
            section_label=section.label,
            pathological=concept.pathological,
            trigger=sentence[trigger.start : trigger.end] if trigger else None,
        )
        if concept.concept == "Steatosis" and status is not Assertion.NEGATED:
            cand.grade = self._grade(sentence, hit.start)
        if load_ontology().is_sizeable(organ.value, concept.finding) and status is not Assertion.NEGATED:
            cand.size_mm = self._size_mm(sentence, hit.start, hit.end)
            cand.location = self._location(sentence, hit.start, hit.end)
        if organ in LATERALISED_ORGANS:
            cand.laterality = self._laterality(sentence, hit.start, hit.end, section)
        return cand

    @staticmethod
    def _dedupe(cands: list[CandidateFinding]) -> list[CandidateFinding]:
        merged: dict[tuple, CandidateFinding] = {}
        for c in cands:
            key = (c.finding, c.status, c.laterality)
            if key not in merged:
                merged[key] = c
                continue
            kept = merged[key]
            if c.size_mm and (kept.size_mm is None or c.size_mm > kept.size_mm):
                kept.size_mm = c.size_mm
            if kept.grade is Grade.NA and c.grade is not Grade.NA:
                kept.grade = c.grade
            kept.location = kept.location or c.location
        return list(merged.values())

    # ------------------------------------------------------------------ normality rules
    def _explicit_unremarkable(
        self, text: str, section: Section, spans: list[tuple[int, int]], organ_findings: list[CandidateFinding]
    ) -> Optional[CandidateFinding]:
        organ = self._organ(section)
        if organ is None or organ is Organ.PELVIS:
            return None
        if any(f.pathological and f.status is not Assertion.NEGATED for f in organ_findings):
            return None
        obscured = any(f.concept == "Bowel Gas Obscuration" and f.status is Assertion.AFFIRMED for f in organ_findings)
        for seg_start, seg_end in spans:
            for s_start, s_end in self._sentences(text, seg_start, seg_end):
                sentence = text[s_start:s_end]
                m = EXPLICIT_UNREMARKABLE.search(sentence)
                if not m:
                    continue
                status, _ = self.context.assert_target(sentence, m.start(), m.end())
                if status is not Assertion.AFFIRMED:
                    continue
                return CandidateFinding(
                    organ=organ,
                    finding="Unremarkable Visualized Portion" if obscured else "Unremarkable",
                    concept="Unremarkable",
                    status=Assertion.AFFIRMED,
                    evidence_text=sentence.strip(),
                    span_start=s_start,
                    span_end=s_end,
                    section_label=section.label,
                    pathological=False,
                )
        return None

    def _study_level(self, text: str, sectioned: SectionedReport, has_pathology: bool) -> Optional[CandidateFinding]:
        if has_pathology:
            return None
        for idx, section in enumerate(sectioned.sections):
            if section.name is not SectionName.IMPRESSION:
                continue
            for seg_start, seg_end in sectioned.segments[idx]:
                for s_start, s_end in self._sentences(text, seg_start, seg_end):
                    sentence = text[s_start:s_end]
                    m = UNREMARKABLE_STUDY.search(sentence)
                    if not m or STUDY_LEVEL_BLOCKERS.search(sentence[: m.start()]):
                        continue
                    return CandidateFinding(
                        organ=Organ.GLOBAL,
                        finding="Unremarkable Study",
                        concept="Unremarkable Study",
                        status=Assertion.AFFIRMED,
                        evidence_text=sentence.strip(),
                        span_start=s_start,
                        span_end=s_end,
                        section_label=section.label,
                        pathological=False,
                    )
        return None

    def _inherited_unremarkable(
        self, text: str, sectioned: SectionedReport, by_section: dict[int, list[CandidateFinding]]
    ) -> list[CandidateFinding]:
        out: list[CandidateFinding] = []
        covered: set[Organ] = set()
        for idx, section in enumerate(sectioned.sections):
            organ = self._organ(section)
            if organ is None or organ not in NORMALITY_DESCRIPTORS or organ in covered:
                continue
            findings = by_section.get(idx, [])
            if any(f.concept == "Unremarkable" for f in findings):
                covered.add(organ)
                continue
            if any(f.pathological and f.status is not Assertion.NEGATED for f in findings):
                continue
            for seg_start, seg_end in sectioned.segments[idx]:
                segment = text[seg_start:seg_end]
                m = NORMALITY_DESCRIPTORS[organ].search(segment)
                if not m:
                    continue
                # evidence = the clause up to the first comma/'and no'/'with no' after the descriptor
                tail = re.search(r",\s*(?:with|and|without)?\s*(?:no|without)\b|\s+(?:and|with)\s+no\b|\s+without\b|,\s+contains\b",
                                 segment[m.end():])
                ev_end = m.end() + (tail.start() if tail else len(segment) - m.end())
                evidence = segment[:ev_end].strip().rstrip(".,")
                out.append(
                    CandidateFinding(
                        organ=organ, finding="Unremarkable", concept="Unremarkable", status=Assertion.AFFIRMED,
                        evidence_text=evidence, span_start=seg_start, span_end=seg_start + ev_end,
                        section_label=section.label, pathological=False,
                        extras={"rule": "inherited-from-study-level-normality"},
                    )
                )
                covered.add(organ)
                break
        return out

    # ------------------------------------------------------------------ public API
    def extract(self, report_id: str, sectioned: SectionedReport) -> tuple[list[Finding], list[str]]:
        """Return validated findings and a list of schema-validation error messages."""
        text = sectioned.text
        by_section: dict[int, list[CandidateFinding]] = {}
        for idx, section in enumerate(sectioned.sections):
            cands = self._extract_section(text, section, sectioned.segments[idx])
            unremarkable = self._explicit_unremarkable(text, section, sectioned.segments[idx], cands)
            if unremarkable:
                cands.append(unremarkable)
            by_section[idx] = cands

        all_cands = [c for cands in by_section.values() for c in cands]
        has_pathology = any(c.pathological and c.status is not Assertion.NEGATED for c in all_cands)
        study = self._study_level(text, sectioned, has_pathology)
        if study:
            all_cands.extend(self._inherited_unremarkable(text, sectioned, by_section))
            all_cands.append(study)

        return self._validate(report_id, all_cands)

    @staticmethod
    def _validate(report_id: str, cands: list[CandidateFinding]) -> tuple[list[Finding], list[str]]:
        from pydantic import ValidationError

        organ_rank = {o: i for i, o in enumerate(Organ)}
        cands.sort(key=lambda c: (organ_rank[c.organ], c.span_start))
        findings: list[Finding] = []
        errors: list[str] = []
        onto = load_ontology()
        for n, c in enumerate(cands, start=1):
            entry = onto.get(c.organ.value, c.finding)
            if entry is not None:
                c.concept = entry.concept
            c.radlex_id, c.icd10_code = onto.codes(c.organ.value, c.finding, c.status.value)
            rule = onto.triage(c.organ.value, c.finding, c.status.value, c.grade.value)
            try:
                findings.append(
                    Finding(
                        report_id=report_id,
                        finding_id=f"{report_id}_P{n:02d}",
                        organ=c.organ,
                        finding=c.finding,
                        concept=c.concept,
                        grade=c.grade,
                        size_mm=c.size_mm,
                        laterality=c.laterality,
                        location=c.location,
                        status=c.status,
                        radlex_id=c.radlex_id,
                        icd10_code=c.icd10_code,
                        evidence_text=c.evidence_text,
                        span_start=c.span_start,
                        span_end=c.span_end,
                        section=c.section_label,
                        triage_tier=TriageTier(rule[0]) if rule else TriageTier.UNREMARKABLE,
                    )
                )
            except ValidationError as exc:
                errors.append(f"{report_id} {c.organ.value}/{c.finding}: {exc.errors()[0]['msg']}")
        return findings, errors
