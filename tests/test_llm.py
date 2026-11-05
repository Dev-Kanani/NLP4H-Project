"""LLM extractor tests with scripted (mock) model responses - no model download needed."""

from __future__ import annotations

import csv
import json
from dataclasses import replace

import pytest

from usg_nlp.config import PipelineConfig
from usg_nlp.deid import PHIScanner
from usg_nlp.evaluation import Evaluator
from usg_nlp.extractors import LLMExtractor, parse_model_json
from usg_nlp.llm import MockBackend, PromptBuilder, ReplayBackend, ResponseCache, select_examples
from usg_nlp.llm.prompts import gold_to_answer
from usg_nlp.ontology import load_ontology
from usg_nlp.pipeline import UltrasoundPipeline
from usg_nlp.verification import FindingVerifier, locate_evidence

REPORT = ("LIVER: Mildly bright liver - Gr I fatty change. No SOL.\n"
          "GB: Multiple calculi, largest 1.4 cm. Wall not thickened.\n"
          "CBD: 4 mm, not dilated.\n"
          "RK: No calculus. PCS not dilated.\n"
          "Imp: Fatty liver gr I. Cholelithiasis.")


def item(organ, finding, status, evidence, grade=None, size=None, lat=None):
    return {"organ": organ, "finding": finding, "status": status, "grade": grade, "size_mm": size,
            "laterality": lat, "evidence": evidence}


# --------------------------------------------------------------------------- parsing & grounding


@pytest.mark.parametrize("raw", [
    '{"findings": []}',
    '```json\n{"findings": []}\n```',
    'Here is the result:\n{"findings": []}\nHope this helps.',
    "[]",
])
def test_parse_model_json_is_tolerant(raw):
    assert parse_model_json(raw) == []


def test_parse_model_json_rejects_garbage():
    with pytest.raises(ValueError):
        parse_model_json("I could not find anything")


def test_locate_evidence_modes():
    assert locate_evidence(REPORT, "No SOL").method == "exact"
    span = locate_evidence(REPORT, "wall  NOT thickened")
    assert span.method == "normalized" and REPORT[span.start:span.end] == "Wall not thickened"
    fuzzy = locate_evidence(REPORT, "Multiple calculi, largest 1.4cm")
    assert fuzzy is not None and fuzzy.method == "fuzzy"
    assert locate_evidence(REPORT, "Large hepatic haemangioma in segment VII") is None


# --------------------------------------------------------------------------- verifier


@pytest.fixture(scope="module")
def verifier():
    return FindingVerifier()


def verify(verifier, items):
    return verifier.verify("R1", REPORT, items, extractor="llm_test")


def test_verifier_normalises_vocabulary_and_organs(verifier):
    res = verify(verifier, [
        item("liver", "fatty liver", "Affirmed", "Gr I fatty change", grade="Grade 1"),
        item("CBD", "CBD Dilatation", "Negated", "not dilated"),
        item("Kidneys", "HDN", "Negated", "PCS not dilated"),
    ])
    got = {(f.organ.value, f.finding, f.status.value) for f in res.findings}
    assert got == {("Liver", "Steatosis", "Affirmed"), ("Gallbladder", "CBD Dilatation", "Negated"),
                   ("Kidneys", "Hydronephrosis", "Negated")}
    steatosis = next(f for f in res.findings if f.finding == "Steatosis")
    assert steatosis.grade.value == "Grade 1" and steatosis.icd10_code == "K76.0" and steatosis.radlex_id == "RID39103"
    assert any(fl.startswith("vocabulary_") for fl in steatosis.flags)


def test_verifier_rejects_hallucinations_and_out_of_vocabulary(verifier):
    res = verify(verifier, [
        item("Liver", "Hemangioma", "Affirmed", "12 mm haemangioma in segment VII"),
        item("Liver", "Cirrhosis", "Affirmed", "Mildly bright liver"),
        item("Gallbladder", "Calculus", "Affirmed", "Multiple calculi, largest 1.4 cm", size="1.4 cm"),
    ])
    reasons = sorted(r["reason"] for r in res.rejected)
    assert reasons == ["out_of_vocabulary", "ungrounded_evidence"]
    stone = res.findings[0]
    assert stone.size_mm == 14.0 and stone.icd10_code == "K80.20"


def test_verifier_attribute_sanity(verifier):
    res = verify(verifier, [
        item("Gallbladder", "Wall Thickening", "Negated", "Wall not thickened", grade="Grade 2", size=3),
        item("Gallbladder", "Calculus", "Affirmed", "Multiple calculi, largest 1.4 cm", size=22),
        item("Gallbladder", "Calculus", "Affirmed", "Multiple calculi, largest 1.4 cm", size=14),
    ])
    wall = next(f for f in res.findings if f.finding == "Wall Thickening")
    assert wall.grade.value == "N/A" and wall.size_mm is None and wall.icd10_code is None
    assert {"grade_dropped", "size_dropped"} <= set(wall.flags)
    stones = [f for f in res.findings if f.finding == "Calculus"]
    assert len(stones) == 1, "exact duplicate span must be dropped"
    assert stones[0].size_mm is None and "size_not_in_text" in stones[0].flags


def test_verifier_flags_assertion_disagreement(verifier):
    res = verify(verifier, [item("Kidneys", "Calculus", "Affirmed", "No calculus")])
    assert res.findings[0].status.value == "Affirmed"  # model output is kept ...
    assert "assertion_disagreement:Negated" in res.findings[0].flags  # ... but flagged for audit


def test_unremarkable_study_maps_to_global(verifier):
    text = "Liver, GB, spleen: NAD.\nImpression: Normal study."
    res = verifier.verify("R2", text, [item("Pelvis", "Unremarkable Study", "Affirmed", "Normal study")], "llm_test")
    f = res.findings[0]
    assert f.organ.value == "Global" and f.icd10_code == "Z00.00"


# --------------------------------------------------------------------------- prompts & extractor


def test_few_shot_examples_never_come_from_test_split():
    cfg = PipelineConfig()
    test_ids = {r["report_id"] for r in csv.DictReader(cfg.splits_csv.open()) if r["split"] == "test"}
    examples = select_examples(cfg.data_dir, n=5)
    assert examples and not ({e.report_id for e in examples} & test_ids)
    assert len({e.style for e in examples[:3]}) == 3


def test_prompt_and_schema_come_from_ontology():
    pb = PromptBuilder()
    schema = pb.output_schema()
    findings_enum = schema["properties"]["findings"]["items"]["properties"]["finding"]["enum"]
    assert "CBD Dilatation" in findings_enum and "Definite Shadowing Calculus" not in findings_enum  # legacy hidden
    assert "Ductal Dilatation" in pb.system_prompt()


def test_prompt_excludes_the_report_being_processed():
    cfg = PipelineConfig()
    ex = select_examples(cfg.data_dir, n=3)
    msgs = PromptBuilder(examples=ex).build(ex[0].text, exclude_report_id=ex[0].report_id)
    assert sum(ex[0].text in m["content"] for m in msgs) == 1  # only as the final user turn


def test_extractor_retries_after_invalid_output():
    answers = iter(["sorry, here you go: {broken", json.dumps({"findings": [item("Liver", "Focal Lesion", "Negated", "No SOL")]})])
    backend = MockBackend(lambda msgs: next(answers))
    ext = LLMExtractor(backend, PromptBuilder(), max_retries=2)
    pipe = UltrasoundPipeline(extractor=ext)
    report, _, errors, diag = pipe.process_text_with_diagnostics("R3", REPORT)
    assert not errors and diag["attempts"] == 2 and len(diag["parse_errors"]) == 1
    assert [(f.finding, f.status.value) for f in report.findings] == [("Focal Lesion", "Negated")]
    assert "not valid" in backend.calls[1][-1]["content"]  # error fed back to the model


# --------------------------------------------------------------------------- end-to-end oracle replay


def _gold_responder(cfg: PipelineConfig):
    onto = load_ontology()
    scanner = PHIScanner()  # the model only ever sees de-identified text
    texts = {}
    for p in cfg.reports_dir.glob("*.txt"):
        texts[scanner.redact(p.stem, p.read_text(encoding="utf-8").strip()).redacted_text] = p.stem
    gold: dict[str, list[dict]] = {}
    for r in csv.DictReader(cfg.gold_csv.open(encoding="utf-8")):
        gold.setdefault(r["report_id"], []).append(r)

    def respond(messages):
        text = messages[-1]["content"].removeprefix("REPORT:\n")
        return json.dumps(gold_to_answer(gold.get(texts[text], []), onto))
    return respond


def test_oracle_replay_end_to_end(tmp_path):
    """A model that answers with the gold labels must score ~1.0: validates prompt -> parse -> verify -> DB -> eval."""
    cfg = replace(PipelineConfig(), output_dir=tmp_path, split="dev", run_name="oracle", bootstrap_resamples=100)
    cache = ResponseCache(tmp_path / "cache.jsonl")
    ext = LLMExtractor(MockBackend(_gold_responder(cfg)), PromptBuilder(examples=select_examples(cfg.data_dir, 2)),
                       cache=cache)
    run = UltrasoundPipeline(cfg, extractor=ext).run(persist=True)
    summary = Evaluator(cfg).evaluate(run).summary
    by_source = {r["source"]: r["f1"] for r in summary["by_source"]}
    # gold answers must survive parsing, grounding (incl. v1's paraphrased spans and legacy names),
    # repeated RK/LK phrases, codes and the evaluator without loss
    assert by_source["v2_generated"] == pytest.approx(1.0) and by_source["original"] == pytest.approx(1.0)
    assert summary["model_reliability"]["hallucination_rate"] == 0.0
    assert (cfg.run_dir / "clinical_ultrasound.db").exists()

    # the cache reproduces the run exactly with no model at all
    replay = ReplayBackend("mock-model", ResponseCache(tmp_path / "cache.jsonl"), replay_identity="mock:mock-model")
    ext2 = LLMExtractor(replay, PromptBuilder(examples=select_examples(cfg.data_dir, 2)))
    run2 = UltrasoundPipeline(replace(cfg, run_name="oracle_replay"), extractor=ext2).run(persist=False)
    key = lambda r: [(f.organ.value, f.finding, f.status.value) for f in r.findings]  # noqa: E731
    assert [key(r) for r in run.reports] == [key(r) for r in run2.reports]
