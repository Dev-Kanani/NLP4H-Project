"""Stage 1b - report and demographic covariate ingestion.

Reads raw ``*.txt`` reports and the cohort metadata CSV, validates every
demographic row through :class:`Demographics`, and cross-checks that the text
on disk matches the copy embedded in the metadata file (catching truncated or
re-exported files before they bias the benchmark).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from .schemas import Demographics

LOGGER = logging.getLogger(__name__)


@dataclass
class RawReport:
    report_id: str
    text: str
    source_path: Path


@dataclass
class IngestionResult:
    reports: list[RawReport]
    demographics: dict[str, Demographics]
    warnings: list[str] = field(default_factory=list)


class ReportIngestor:
    def __init__(self, reports_dir: Path, metadata_csv: Path | None = None) -> None:
        self.reports_dir = reports_dir
        self.metadata_csv = metadata_csv

    def load_reports(self) -> list[RawReport]:
        if not self.reports_dir.is_dir():
            raise FileNotFoundError(f"Reports directory not found: {self.reports_dir}")
        reports = [
            RawReport(report_id=path.stem, text=path.read_text(encoding="utf-8").strip(), source_path=path)
            for path in sorted(self.reports_dir.glob("*.txt"))
        ]
        if not reports:
            raise FileNotFoundError(f"No .txt reports in {self.reports_dir}")
        return reports

    def load_demographics(self) -> tuple[dict[str, Demographics], pd.DataFrame | None, list[str]]:
        warnings: list[str] = []
        if self.metadata_csv is None or not self.metadata_csv.exists():
            warnings.append("Metadata CSV not supplied; demographic analytics disabled.")
            return {}, None, warnings
        frame = pd.read_csv(self.metadata_csv)
        demographics: dict[str, Demographics] = {}
        for row in frame.to_dict(orient="records"):
            try:
                demo = Demographics(
                    report_id=str(row["report_id"]),
                    age=int(row["age"]),
                    sex=str(row["sex"]).strip().title(),
                    style=str(row["style"]).strip().title(),
                )
            except (ValidationError, KeyError, ValueError) as exc:
                warnings.append(f"Rejected metadata row {row.get('report_id')}: {exc}")
                continue
            demographics[demo.report_id] = demo
        return demographics, frame, warnings

    def ingest(self) -> IngestionResult:
        reports = self.load_reports()
        demographics, frame, warnings = self.load_demographics()
        if frame is not None and "report_text" in frame.columns:
            embedded = dict(zip(frame["report_id"].astype(str), frame["report_text"].astype(str)))
            for report in reports:
                ref = embedded.get(report.report_id)
                if ref is not None and " ".join(ref.split()) != " ".join(report.text.split()):
                    warnings.append(f"{report.report_id}: text on disk differs from metadata copy")
        for report in reports:
            if demographics and report.report_id not in demographics:
                warnings.append(f"{report.report_id}: no demographic metadata")
        for message in warnings:
            LOGGER.warning(message)
        return IngestionResult(reports=reports, demographics=demographics, warnings=warnings)
