"""Unit tests for de-identification, sectionization, ConText and schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from usg_nlp.context import ConTextEngine
from usg_nlp.deid import PHIScanner
from usg_nlp.schemas import Assertion, Finding, Grade, Laterality, Organ, SectionName
from usg_nlp.sectionizer import AnatomicalSectionizer

# --------------------------------------------------------------------------- Stage 1


class TestDeidentification:
    scanner = PHIScanner()

    def test_detects_direct_identifiers(self):
        text = ("Patient Name: Ramesh Kumar   MRN: HYD-2024-88123   Date: 12/03/2024\n"
                "Ref by: Dr. Anita Rao  Ph: +91 98480 22334  email r.kumar@example.com  Age: 93 years")
        types = [e.phi_type for e in self.scanner.detect(text)]
        for expected in ("NAME", "MRN", "DATE", "PHONE", "EMAIL", "AGE_OVER_89"):
            assert expected in types
        redacted = self.scanner.redact("X", text).redacted_text
        for leaked in ("Ramesh", "Kumar", "88123", "12/03/2024", "98480", "Anita", "Rao", "example.com"):
            assert leaked not in redacted

    def test_clinical_measurements_are_not_phi(self):
        text = "Simple cortical cyst, lower pole, 8.0 mm. Calculus 12.4 mm. Grade 1 steatosis. Liver 15.2 cm."
        assert self.scanner.detect(text) == []

    def test_month_name_dates(self):
        assert {e.phi_type for e in self.scanner.detect("Scanned on 12 March 2024 and March 14, 2024.")} == {"DATE"}


# --------------------------------------------------------------------------- Stage 2


class TestSectionizer:
    sz = AnatomicalSectionizer()

    def test_structured_headers(self):
        text = "LIVER: Normal.\n\nGALLBLADDER: No calculus.\n\nKIDNEYS: No hydronephrosis.\n\nIMPRESSION: Normal study."
        rep = self.sz.sectionize(text)
        assert rep.style_detected == "Structured"
        assert [s.name for s in rep.sections] == [SectionName.LIVER, SectionName.GALLBLADDER,
                                                  SectionName.KIDNEYS, SectionName.IMPRESSION]
        assert all(s.method == "header" for s in rep.sections)

    def test_telegraphic_abbreviations_laterality_and_combined_header(self):
        text = ("Liver: WNL.\nGB: No stones.\nPAN/SPL: WNL; no focal abnormality. No focal splenic lesion.\n"
                "RK: Simple cortical cyst, 9 mm.\nLK: No calculus.\nImp: Right renal cyst.")
        rep = self.sz.sectionize(text)
        assert rep.style_detected == "Telegraphic"
        labels = [s.label for s in rep.sections]
        assert "Kidneys (Right)" in labels and "Kidneys (Left)" in labels
        spleen = next(s for s in rep.sections if s.name is SectionName.SPLEEN)
        pancreas = next(s for s in rep.sections if s.name is SectionName.PANCREAS)
        assert "splenic" in spleen.text and "splenic" not in pancreas.text

    def test_unstructured_clause_routing_and_inheritance(self):
        text = ("The liver is normal in size, and the gallbladder contains no calculi. Both kidneys are normal; "
                "however, a tiny left renal calculus cannot be entirely excluded. The spleen is unremarkable, "
                "and no free pelvic fluid is identified. Overall, this is an unremarkable study.")
        rep = self.sz.sectionize(text)
        assert rep.style_detected == "Unstructured"
        by_name = {s.name: s for s in rep.sections}
        assert "tiny left renal calculus" in by_name[SectionName.KIDNEYS].text  # inherited, anchor-less clause
        assert "free pelvic fluid" in by_name[SectionName.PELVIS].text
        assert "Overall" in by_name[SectionName.IMPRESSION].text
        assert "gallbladder" not in by_name[SectionName.LIVER].text

    def test_segments_point_at_source_offsets(self):
        text = "The liver is normal, and the gallbladder is thin-walled. No hydronephrosis in either kidney."
        rep = self.sz.sectionize(text)
        for section, start, end in rep.iter_segments():
            assert text[start:end] in section.text


# --------------------------------------------------------------------------- Stage 3: ConText


class TestConText:
    ctx = ConTextEngine()

    def _status(self, sentence: str, target: str) -> Assertion:
        start = sentence.index(target)
        return self.ctx.assert_target(sentence, start, start + len(target))[0]

    @pytest.mark.parametrize("sentence,target,expected", [
        ("No calculus, wall thickening, or pericholecystic fluid.", "pericholecystic fluid", Assertion.NEGATED),
        ("The intrahepatic bile ducts are not dilated.", "intrahepatic bile ducts", Assertion.NEGATED),
        ("Sonographic Murphy sign is negative.", "Murphy sign", Assertion.NEGATED),
        ("A tiny left renal calculus cannot be entirely excluded.", "renal calculus", Assertion.POSSIBLE),
        ("Calculus cannot be ruled out due to obscuring bowel gas.", "Calculus", Assertion.POSSIBLE),
        ("Possible small polyp along the fundus.", "polyp", Assertion.POSSIBLE),
        ("Simple cortical cyst, lower pole, 8.0 mm.", "cortical cyst", Assertion.AFFIRMED),
        ("No significant change in the hepatic cyst.", "hepatic cyst", Assertion.AFFIRMED),  # pseudo-trigger
        ("No hydronephrosis but a renal calculus is seen.", "renal calculus", Assertion.AFFIRMED),  # terminator
    ])
    def test_assertion(self, sentence, target, expected):
        assert self._status(sentence, target) is expected

    def test_nearest_trigger_wins_in_hedged_sentence(self):
        s = "although no definite shadowing calculus is demonstrated, a tiny left renal calculus cannot be entirely excluded"
        assert self._status(s, "definite shadowing calculus") is Assertion.NEGATED
        assert self._status(s, "renal calculus") is Assertion.POSSIBLE

    def test_scope_limit(self):
        s = "No focal lesion is seen in the liver which otherwise demonstrates a large and very echogenic cyst"
        assert self._status(s, "cyst") is Assertion.AFFIRMED

    def test_medspacy_export(self):
        rules = self.ctx.to_medspacy_rules()
        assert {"literal", "category", "direction"} <= set(rules[0])
        assert any(r["category"] == "NEGATED_EXISTENCE" for r in rules)


# --------------------------------------------------------------------------- Stage 4: schema


class TestSchema:
    base = dict(report_id="R1", finding_id="R1_P01", organ=Organ.LIVER, finding="Steatosis", concept="Steatosis",
                status=Assertion.AFFIRMED, evidence_text="Grade 1 steatosis", span_start=0, span_end=10, section="Liver")

    def test_valid_finding(self):
        f = Finding(**self.base, grade=Grade.GRADE_1, radlex_id="RID39103", icd10_code="K76.0")
        assert f.grade is Grade.GRADE_1

    def test_icd_not_allowed_on_negated(self):
        with pytest.raises(ValidationError):
            Finding(**{**self.base, "status": Assertion.NEGATED}, icd10_code="K76.0")

    def test_grade_only_for_steatosis(self):
        with pytest.raises(ValidationError):
            Finding(**{**self.base, "finding": "Calculus", "concept": "Calculus"}, grade=Grade.GRADE_2)

    @pytest.mark.parametrize("field,value", [("radlex_id", "R39103"), ("icd10_code", "K760"), ("size_mm", -2.0)])
    def test_malformed_values_rejected(self, field, value):
        with pytest.raises(ValidationError):
            Finding(**self.base, **{field: value})

    def test_laterality_enum(self):
        f = Finding(**{**self.base, "organ": Organ.KIDNEYS, "finding": "Cortical Cyst", "concept": "Cortical Cyst"},
                    laterality=Laterality.LEFT, size_mm=12.0)
        assert f.laterality is Laterality.LEFT
