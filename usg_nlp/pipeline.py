"""End-to-end orchestration of Stages 1-4 with a pluggable extractor.

``UltrasoundPipeline.process_text`` is the single-report entry point (used by
the dashboard's live structurer); ``UltrasoundPipeline.run`` processes the
cohort (or one split of it) and persists every artefact under
``outputs/<run_name>/``.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from . import __version__
from .config import PipelineConfig
from .database import ClinicalDatabase
from .deid import PHIScanner
from .extractors import BaseExtractor, RuleBasedExtractor
from .ingestion import ReportIngestor
from .schemas import Demographics, StructuredReport
from .sectionizer import AnatomicalSectionizer
from .triage import FollowUpClassifier

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineRun:
    reports: list[StructuredReport]
    raw_texts: dict[str, str]
    style_detected: dict[str, str]
    schema_errors: list[str]
    ingestion_warnings: list[str]
    nlp_seconds: float
    extractor_info: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_seconds: float = 0.0

    @property
    def n_findings(self) -> int:
        return sum(len(r.findings) for r in self.reports)

    @property
    def schema_validity_pass_rate(self) -> float:
        total = self.n_findings + len(self.schema_errors)
        return self.n_findings / total if total else 1.0

    @property
    def throughput(self) -> float:
        return len(self.reports) / self.nlp_seconds if self.nlp_seconds else float("inf")

    def findings_frame(self) -> pd.DataFrame:
        rows = [f.model_dump(mode="json") for r in self.reports for f in r.findings]
        return pd.DataFrame(rows)

    def reliability(self) -> dict[str, float]:
        """Model-output reliability: proposals, rejections by reason, parse failures."""
        proposed = sum(d.get("proposed", 0) for d in self.diagnostics.values())
        rejected = [r for d in self.diagnostics.values() for r in d.get("rejected", [])]
        reasons: dict[str, int] = {}
        for r in rejected:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        flags: dict[str, int] = {}
        for rep in self.reports:
            for f in rep.findings:
                for fl in f.flags:
                    k = "evidence_fuzzy" if fl.startswith("evidence_fuzzy") else fl.split(":")[0]
                    flags[k] = flags.get(k, 0) + 1
        failed = sum(1 for d in self.diagnostics.values() if d.get("backend_error") or
                     (d.get("attempts", 0) and d.get("proposed", 0) == 0 and d.get("parse_errors")))
        return {
            "proposed": proposed,
            "accepted": self.n_findings,
            "rejected_by_reason": reasons,
            "hallucination_rate": reasons.get("ungrounded_evidence", 0) / proposed if proposed else 0.0,
            "out_of_vocabulary_rate": reasons.get("out_of_vocabulary", 0) / proposed if proposed else 0.0,
            "reports_without_parsable_output": failed,
            "retries": sum(max(0, d.get("attempts", 1) - 1) for d in self.diagnostics.values()),
            "flags": flags,
        }


class UltrasoundPipeline:
    def __init__(self, config: PipelineConfig | None = None, extractor: Optional[BaseExtractor] = None,
                 workers: int = 1) -> None:
        self.config = config or PipelineConfig()
        self.scanner = PHIScanner()
        self.sectionizer = AnatomicalSectionizer()
        self.extractor = extractor or RuleBasedExtractor()
        self.classifier = FollowUpClassifier()
        self.workers = max(1, workers)

    # ------------------------------------------------------------------ single report
    def process_text(self, report_id: str, text: str, demographics: Demographics | None = None,
                     ) -> tuple[StructuredReport, str, list[str]]:
        report, style, errors, _ = self.process_text_with_diagnostics(report_id, text, demographics)
        return report, style, errors

    def process_text_with_diagnostics(self, report_id: str, text: str, demographics: Demographics | None = None):
        t0 = time.perf_counter()
        deid = self.scanner.redact(report_id, text)
        sectioned = self.sectionizer.sectionize(deid.redacted_text)
        result = self.extractor.extract(report_id, deid.redacted_text, sectioned)
        tier, reasons = self.classifier.classify(result.findings)
        report = StructuredReport(
            report_id=report_id,
            demographics=demographics,
            deidentification=deid,
            sections=sectioned.sections,
            findings=result.findings,
            conditions=self.classifier.conditions(result.findings),
            triage_tier=tier,
            triage_reasons=reasons,
            processing_ms=(time.perf_counter() - t0) * 1000.0,
        )
        return report, sectioned.style_detected, result.schema_errors, result.diagnostics

    # ------------------------------------------------------------------ cohort
    def run(self, persist: bool = True) -> PipelineRun:
        t_total = time.perf_counter()
        ingestion = ReportIngestor(self.config.reports_dir, self.config.metadata_csv).ingest()
        keep = self.config.report_ids_for_split()
        raws = [r for r in ingestion.reports if keep is None or r.report_id in keep]

        t_nlp = time.perf_counter()

        def one(raw):
            return raw, self.process_text_with_diagnostics(raw.report_id, raw.text,
                                                           ingestion.demographics.get(raw.report_id))

        if self.workers > 1:
            with ThreadPoolExecutor(self.workers) as pool:
                outputs = list(pool.map(one, raws))
        else:
            outputs = [one(r) for r in raws]
        nlp_seconds = time.perf_counter() - t_nlp

        run = PipelineRun(
            reports=[o[1][0] for o in outputs],
            raw_texts={raw.report_id: raw.text for raw, _ in outputs},
            style_detected={raw.report_id: o[1] for raw, o in outputs},
            schema_errors=[e for _, o in outputs for e in o[2]],
            ingestion_warnings=ingestion.warnings,
            nlp_seconds=nlp_seconds,
            extractor_info=self.extractor.describe(),
            diagnostics={raw.report_id: o[3] for raw, o in outputs},
        )
        if persist:
            self.persist(run)
        run.total_seconds = time.perf_counter() - t_total
        LOGGER.info("Processed %d reports (%d findings) in %.3fs NLP / %.3fs total",
                    len(run.reports), run.n_findings, nlp_seconds, run.total_seconds)
        return run

    def persist(self, run: PipelineRun) -> None:
        cfg = self.config
        cfg.ensure_output_dir()
        run.findings_frame().to_csv(cfg.predictions_csv, index=False)
        cfg.predictions_json.write_text(
            json.dumps([r.model_dump(mode="json") for r in run.reports], indent=2), encoding="utf-8"
        )
        (cfg.run_dir / "model_diagnostics.json").write_text(
            json.dumps({"extractor": run.extractor_info, "reliability": run.reliability(),
                        "per_report": run.diagnostics}, indent=2, default=str), encoding="utf-8")
        audit_rows = [
            {"report_id": r.report_id, "phi_count": r.deidentification.phi_count,
             "phi_types": ";".join(sorted({e.phi_type for e in r.deidentification.phi_entities})),
             "safe_harbor_compliant_at_source": r.deidentification.safe_harbor_compliant_at_source}
            for r in run.reports
        ]
        pd.DataFrame(audit_rows).to_csv(cfg.phi_audit_csv, index=False)
        metadata = {
            "pipeline_version": __version__,
            "run_name": cfg.run_name,
            "split": cfg.split,
            "n_reports": str(len(run.reports)),
            "n_findings": str(run.n_findings),
            "schema_validity_pass_rate": f"{run.schema_validity_pass_rate:.4f}",
            "nlp_seconds": f"{run.nlp_seconds:.4f}",
            **{f"extractor_{k}": str(v) for k, v in run.extractor_info.items()},
            **{f"style_detected:{k}": v for k, v in run.style_detected.items()},
        }
        ClinicalDatabase(cfg.db_path).write(run.reports, run.raw_texts, metadata)
