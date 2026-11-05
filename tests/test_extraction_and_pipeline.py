"""Extraction behaviour on unseen phrasings, plus end-to-end regression gates on the cohort."""

from __future__ import annotations

import sqlite3

import pytest

from usg_nlp.evaluation import Evaluator
from usg_nlp.schemas import Grade, Laterality, TriageTier

from .conftest import findings_of

# --------------------------------------------------------------------------- unseen phrasings
# None of these sentences occur in the 50-report corpus; they exercise synonymy and
# formatting variants named in the proposal (Section 2.3).


def test_telegraphic_shorthand_nad_and_roman_grade(pipeline):
    report, style, _ = pipeline.process_text(
        "T", "Liver: Grade-II fatty liver. Hepatomegaly.\nGB: NAD\nPan: Obscured by bowel gas.\n"
             "KUB: WNL. No calculus/hydronephrosis.\nImp: Grade II fatty liver with hepatomegaly.")
    found = {(f.organ.value, f.finding, f.status.value): f for f in report.findings}
    assert style == "Telegraphic"
    assert found[("Liver", "Steatosis", "Affirmed")].grade is Grade.GRADE_2
    assert ("Liver", "Hepatomegaly", "Affirmed") in found
    assert ("Gallbladder", "Unremarkable", "Affirmed") in found
    assert ("Pancreas", "Bowel Gas Obscuration", "Affirmed") in found
    assert {("Kidneys", "Calculus", "Negated"), ("Kidneys", "Hydronephrosis", "Negated")} <= set(found)
    assert report.triage_tier is TriageTier.ROUTINE_TRACKING


def test_descriptive_steatosis_and_hedged_gallstone(pipeline):
    found = findings_of(pipeline,
        "LIVER: Diffuse increase in hepatic parenchymal echogenicity. No focal solid or cystic mass detected.\n"
        "GALLBLADDER: Calculus cannot be ruled out due to obscuring bowel gas.\n"
        "KIDNEYS: Right kidney shows a 4.2 mm calculus at the lower pole without hydronephrosis.\n"
        "IMPRESSION: Fatty liver.")
    assert {("Liver", "Steatosis", "Affirmed"), ("Liver", "Focal Lesion", "Negated"),
            ("Gallbladder", "Calculus", "Possible"), ("Kidneys", "Calculus", "Affirmed"),
            ("Kidneys", "Hydronephrosis", "Negated")} <= found


def test_narrative_units_laterality_and_routing(pipeline):
    report, style, _ = pipeline.process_text(
        "N", "Mild steatosis hepatis is seen with no evidence of focal lesion. The gall bladder shows a 0.8 cm "
             "polyp along the fundus. A 1.2 x 1.0 cm simple cyst is noted in the upper pole of the left kidney. "
             "No ascites.")
    by_finding = {f.finding: f for f in report.findings}
    assert style == "Unstructured"
    assert by_finding["Steatosis"].grade is Grade.GRADE_1
    assert by_finding["Polyp"].size_mm == 8.0
    cyst = by_finding["Cortical Cyst"]
    assert (cyst.size_mm, cyst.laterality, cyst.location) == (12.0, Laterality.LEFT, "upper pole")
    assert by_finding["Free Fluid"].organ.value == "Pelvis"


def test_otherwise_unremarkable_is_not_a_normal_study(pipeline):
    found = findings_of(pipeline, "R kidney: Simple cortical cyst, 9.0 mm.\nL kidney: WNL.\n"
                                  "Imp: Small right renal cyst. Otherwise unremarkable abdominal US.")
    assert not any(f[1] == "Unremarkable Study" for f in found)


def test_normal_study_grounded_to_z00(pipeline):
    report, _, _ = pipeline.process_text("Z", "LIVER: Normal size.\nGB: No stones.\nIMPRESSION: Normal study.")
    study = [f for f in report.findings if f.finding == "Unremarkable Study"]
    assert study and study[0].icd10_code == "Z00.00" and study[0].radlex_id == "RID28473"
    assert report.triage_tier is TriageTier.UNREMARKABLE


def test_grade3_steatosis_escalates_to_review(pipeline):
    report, _, _ = pipeline.process_text("G3", "LIVER: Markedly hyperechoic liver, Grade 3 steatosis.\n"
                                               "GB: No calculus.")
    assert report.triage_tier is TriageTier.CLINICIAN_REVIEW


# --------------------------------------------------------------------------- end-to-end gates


def test_cohort_runs_fast_and_validates(cohort_run, tmp_config):
    import csv
    assert len(cohort_run.reports) == 75  # dev split
    assert cohort_run.nlp_seconds < 5.0, "CPU budget for the rule baseline: < 5 s"
    assert cohort_run.schema_validity_pass_rate == 1.0
    with_phi = {r["report_id"] for r in csv.DictReader(tmp_config.phi_gold_csv.open())}
    for r in cohort_run.reports:  # identifiers are found exactly where the PHI gold says they are (dev reports)
        assert (r.deidentification.phi_count > 0) == (r.report_id in with_phi), r.report_id


def test_database_integrity(cohort_run, tmp_config):
    with sqlite3.connect(tmp_config.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0] == 75
        assert conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0] == cohort_run.n_findings
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert {"ix_findings_concept", "ix_patients_band_sex", "ix_findings_extractor"} <= indexes
        prevalence = dict(conn.execute("SELECT condition, n FROM v_condition_prevalence").fetchall())
        assert prevalence["Hepatic steatosis"] == sum(r.conditions.steatosis for r in cohort_run.reports)
        assert conn.execute("SELECT DISTINCT extractor FROM findings").fetchall() == [("rules",)]


@pytest.fixture(scope="module")
def evaluation(cohort_run, tmp_config):
    return Evaluator(tmp_config).evaluate(cohort_run)


def test_benchmark_regression_gate(evaluation):
    """Frozen rule baseline: guards against accidental behaviour changes during refactors."""
    s = evaluation.summary
    by_source = {r["source"]: r for r in s["by_source"]}
    assert by_source["original"]["f1"] >= 0.99  # the data its rules were written on
    assert 0.6 <= by_source["v2_generated"]["f1"] <= 0.8  # unseen phrasing: the documented generalisation gap
    assert s["phi"]["recall"] == 1.0 and s["phi"]["false_positives"] == 0
    assert s["splits_evaluated"] == ["dev"]
    lo, hi = s["entity_concept"]["ci95"]["f1"]
    assert lo <= s["entity_concept"]["f1"] <= hi
    assert s["size_mm"]["mae_mm"] < 0.5
    assert s["steatosis_grade"]["accuracy"] >= 0.95
    assert s["hallucination_rate"] == 0.0
    assert {row["style"] for row in s["by_style"]} == {"Structured", "Telegraphic", "Unstructured"}


def test_multiset_matching_counts_duplicates(tmp_config):
    import pandas as pd

    from usg_nlp.evaluation import multiset_counts

    gold = pd.DataFrame({"report_id": ["R"] * 2, "organ_norm": ["Kidneys"] * 2,
                         "concept": ["Hydronephrosis"] * 2, "status": ["Negated"] * 2})
    pred = gold.iloc[:1]
    key = lambda r: (r["organ_norm"], r["concept"], r["status"])  # noqa: E731
    t = multiset_counts(gold, pred, ["R"], key)
    assert (t.tp.sum(), t.fp.sum(), t.fn.sum()) == (1, 0, 1)
