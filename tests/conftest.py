from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from usg_nlp.config import PipelineConfig  # noqa: E402
from usg_nlp.pipeline import UltrasoundPipeline  # noqa: E402


@pytest.fixture(scope="session")
def pipeline() -> UltrasoundPipeline:
    return UltrasoundPipeline()


@pytest.fixture(scope="session")
def tmp_config(tmp_path_factory) -> PipelineConfig:
    # tests never touch the locked test split
    return replace(PipelineConfig(), output_dir=tmp_path_factory.mktemp("outputs"), bootstrap_resamples=200, split="dev")


@pytest.fixture(scope="session")
def cohort_run(tmp_config):
    return UltrasoundPipeline(tmp_config).run(persist=True)


def findings_of(pipeline: UltrasoundPipeline, text: str) -> set[tuple[str, str, str]]:
    report, _, errors = pipeline.process_text("TEST_001", text)
    assert not errors, errors
    return {(f.organ.value, f.finding, f.status.value) for f in report.findings}
