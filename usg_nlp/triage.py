"""Stage 4a - report-level condition flags and 3-tier follow-up grouping.

The report tier is the maximum tier over its findings.  Categories are
assistive administrative groupings (proposal Section 5); they are *not*
autonomous clinical triage or diagnosis.
"""

from __future__ import annotations

from .ontology import load_ontology
from .schemas import Assertion, Finding, Grade, Organ, ReportConditions, TriageTier

_GRADE_ORDER = {Grade.NA: 0, Grade.GRADE_1: 1, Grade.GRADE_2: 2, Grade.GRADE_3: 3}


class FollowUpClassifier:
    def conditions(self, findings: list[Finding]) -> ReportConditions:
        affirmed = [f for f in findings if f.status is Assertion.AFFIRMED]
        steatosis = [f for f in affirmed if f.concept == "Steatosis"]
        grade = max((f.grade for f in steatosis), key=_GRADE_ORDER.__getitem__, default=Grade.NA)
        return ReportConditions(
            steatosis=bool(steatosis),
            steatosis_grade=grade,
            hepatomegaly=any(f.concept == "Hepatomegaly" for f in affirmed),
            cholelithiasis=any(f.concept == "Calculus" and f.organ is Organ.GALLBLADDER for f in affirmed),
            renal_cyst=any(f.concept == "Cortical Cyst" for f in affirmed),
            renal_calculus=any(f.concept == "Calculus" and f.organ is Organ.KIDNEYS for f in affirmed),
            possible_renal_calculus=any(
                f.concept == "Calculus" and f.organ is Organ.KIDNEYS and f.status is Assertion.POSSIBLE for f in findings
            ),
            any_abnormality=any(
                load_ontology().is_pathological(f.organ.value, f.finding) and f.status is not Assertion.NEGATED
                for f in findings
            ),
        )

    def classify(self, findings: list[Finding]) -> tuple[TriageTier, list[str]]:
        tier = TriageTier.UNREMARKABLE
        reasons: list[str] = []
        onto = load_ontology()
        for f in findings:
            rule = onto.triage(f.organ.value, f.finding, f.status.value, f.grade.value)
            if rule is None:
                continue
            size = f" ({f.size_mm:g} mm)" if f.size_mm else ""
            reason = f"{f.organ.value}: {rule[1]}{size}"
            if reason not in reasons:
                reasons.append(reason)
            tier = max(tier, TriageTier(rule[0]))
        if tier is TriageTier.UNREMARKABLE:
            reasons.append("No affirmed or indeterminate pathology")
        if any(f.concept == "Bowel Gas Obscuration" for f in findings):
            reasons.append("Quality note: partial obscuration by bowel gas")
        return tier, reasons
