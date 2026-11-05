"""Population health analytics on top of the SQLite store (shared by dashboard and notebooks).

All functions return tidy DataFrames.  Prevalence is always reported with its
numerator and denominator, and with a Wilson 95% interval, because subgroup
cells in a 50-person cohort are small: an "18% vs 40%" contrast between two
age bands may rest on a handful of people.
"""

from __future__ import annotations

import math

import pandas as pd

from .config import AGE_BANDS
from .database import ClinicalDatabase

CONDITIONS: dict[str, str] = {
    "steatosis": "Hepatic steatosis",
    "cholelithiasis": "Cholelithiasis",
    "renal_cyst": "Renal cortical cyst",
    "hepatomegaly": "Hepatomegaly",
    "possible_renal_calculus": "Possible renal calculus",
    "any_abnormality": "Any abnormality",
}
AGE_BAND_ORDER = [label for label, _, _ in AGE_BANDS]


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


class PopulationAnalytics:
    def __init__(self, db: ClinicalDatabase) -> None:
        self.db = db

    def cohort(self) -> pd.DataFrame:
        frame = self.db.table("v_patient_conditions")
        frame["age_band"] = pd.Categorical(frame["age_band"], AGE_BAND_ORDER, ordered=True)
        return frame

    def findings(self) -> pd.DataFrame:
        return self.db.table("findings")

    @staticmethod
    def prevalence(cohort: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
        rows = []
        groups = [("All", cohort)] if by is None else list(cohort.groupby(by, observed=True))
        for key, grp in groups:
            n = len(grp)
            for col, label in CONDITIONS.items():
                k = int(grp[col].sum())
                lo, hi = wilson_interval(k, n)
                rows.append({"group": key, "condition": label, "cases": k, "n": n,
                             "prevalence": k / n if n else float("nan"), "ci_low": lo, "ci_high": hi})
        return pd.DataFrame(rows)

    @staticmethod
    def crosstab(cohort: pd.DataFrame, condition: str) -> pd.DataFrame:
        """Cases / n per age band x sex for one condition column."""
        cases = pd.crosstab(cohort["age_band"], cohort["sex"], values=cohort[condition], aggfunc="sum").fillna(0)
        totals = pd.crosstab(cohort["age_band"], cohort["sex"])
        out = cases.astype(int).astype(str) + " / " + totals.astype(str)
        return out.reindex([b for b in AGE_BAND_ORDER if b in out.index])

    @staticmethod
    def co_occurrence(cohort: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in CONDITIONS if c != "any_abnormality"]
        mat = pd.DataFrame(index=[CONDITIONS[c] for c in cols], columns=[CONDITIONS[c] for c in cols], dtype=int)
        for a in cols:
            for b in cols:
                mat.loc[CONDITIONS[a], CONDITIONS[b]] = int((cohort[a].astype(bool) & cohort[b].astype(bool)).sum())
        return mat

    def steatosis_grades(self, cohort: pd.DataFrame) -> pd.DataFrame:
        sub = cohort[cohort["steatosis"] == 1]
        counts = sub["steatosis_grade"].value_counts().reindex(["Grade 1", "Grade 2", "Grade 3", "N/A"]).fillna(0)
        return counts.astype(int).rename_axis("grade").reset_index(name="patients")
