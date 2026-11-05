"""Central configuration: filesystem layout and global constants.

Every path can be overridden through environment variables so the same
package runs unchanged on a laptop, the Ada cluster, or inside CI.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable run configuration passed explicitly through every stage."""

    data_dir: Path = field(default_factory=lambda: _env_path("USG_DATA_DIR", PROJECT_ROOT / "data"))
    output_dir: Path = field(default_factory=lambda: _env_path("USG_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 20261029
    confidence_level: float = 0.95
    run_name: str = "rules"  # each extractor/model writes to outputs/<run_name>/
    split: str = "all"  # which reports to process: all | dev | test (see data/splits.csv)

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def metadata_csv(self) -> Path:
        return self.data_dir / "reports_metadata.csv"

    @property
    def gold_csv(self) -> Path:
        return self.data_dir / "ground_truth_findings.csv"

    @property
    def splits_csv(self) -> Path:
        return self.data_dir / "splits.csv"

    @property
    def phi_gold_csv(self) -> Path:
        return self.data_dir / "phi_ground_truth.csv"

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name

    @property
    def llm_cache_dir(self) -> Path:
        return self.output_dir / "llm_cache"

    @property
    def test_log(self) -> Path:
        return self.output_dir / "test_set_evaluations.log"

    @property
    def db_path(self) -> Path:
        return self.run_dir / "clinical_ultrasound.db"

    @property
    def predictions_csv(self) -> Path:
        return self.run_dir / "predicted_findings.csv"

    @property
    def predictions_json(self) -> Path:
        return self.run_dir / "structured_reports.json"

    @property
    def phi_audit_csv(self) -> Path:
        return self.run_dir / "phi_audit.csv"

    @property
    def evaluation_json(self) -> Path:
        return self.run_dir / "evaluation_summary.json"

    @property
    def error_analysis_csv(self) -> Path:
        return self.run_dir / "error_analysis.csv"

    def ensure_output_dir(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def report_ids_for_split(self) -> Optional[set[str]]:
        """Report ids in ``self.split``; ``None`` means every report."""
        if self.split == "all":
            return None
        if not self.splits_csv.exists():
            raise FileNotFoundError(f"--split {self.split} needs {self.splits_csv}")
        with self.splits_csv.open(encoding="utf-8") as fh:
            return {r["report_id"] for r in csv.DictReader(fh) if r["split"] == self.split}


# Canonical organ order used by the sectionizer, the UI and the database.
ORGANS: tuple[str, ...] = ("Liver", "Gallbladder", "Pancreas", "Spleen", "Kidneys", "Pelvis")
SECTIONS: tuple[str, ...] = ORGANS + ("Impression",)

# Age bands follow the proposal (Section 3.6): <30, 30-39, 40-49, 50+.
AGE_BANDS: tuple[tuple[str, int, int], ...] = (
    ("<30", 0, 29),
    ("30-39", 30, 39),
    ("40-49", 40, 49),
    ("50+", 50, 200),
)


def age_band(age: int) -> str:
    for label, lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return label
    raise ValueError(f"Age {age} outside configured bands")
