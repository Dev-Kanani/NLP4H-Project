"""Pydantic data contracts for every artefact the pipeline produces.

These models are the single source of truth for typing.  Every finding the
extractor emits is validated here before it reaches SQLite, so a malformed
record fails loudly instead of silently corrupting population statistics.
The Schema Validity Pass Rate (SVPR) reported by the evaluator is literally
the fraction of candidate records that pass these validators.
"""

from __future__ import annotations

import re
from enum import Enum, IntEnum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from .config import age_band

REPORT_ID_PATTERN = r"^[A-Za-z0-9_\-]+$"
RADLEX_PATTERN = re.compile(r"^RID\d{2,7}$")
ICD10_PATTERN = re.compile(r"^[A-TV-Z]\d{2}(?:\.[0-9A-Z]{1,4})?$")


class Organ(str, Enum):
    LIVER = "Liver"
    GALLBLADDER = "Gallbladder"
    PANCREAS = "Pancreas"
    SPLEEN = "Spleen"
    KIDNEYS = "Kidneys"
    PELVIS = "Pelvis"
    GLOBAL = "Global"  # study-level statements, e.g. "unremarkable abdominal ultrasound study"


class SectionName(str, Enum):
    LIVER = "Liver"
    GALLBLADDER = "Gallbladder"
    PANCREAS = "Pancreas"
    SPLEEN = "Spleen"
    KIDNEYS = "Kidneys"
    PELVIS = "Pelvis"
    IMPRESSION = "Impression"


class Assertion(str, Enum):
    AFFIRMED = "Affirmed"
    NEGATED = "Negated"
    POSSIBLE = "Possible"


class Grade(str, Enum):
    GRADE_1 = "Grade 1"
    GRADE_2 = "Grade 2"
    GRADE_3 = "Grade 3"
    NA = "N/A"


class Laterality(str, Enum):
    LEFT = "Left"
    RIGHT = "Right"
    BILATERAL = "Bilateral"


class TriageTier(IntEnum):
    UNREMARKABLE = 1
    ROUTINE_TRACKING = 2
    CLINICIAN_REVIEW = 3

    @property
    def label(self) -> str:
        return {
            TriageTier.UNREMARKABLE: "Unremarkable Findings",
            TriageTier.ROUTINE_TRACKING: "Routine Preventive Tracking",
            TriageTier.CLINICIAN_REVIEW: "Findings Warranting Clinician Review",
        }[self]


class ReportingStyle(str, Enum):
    STRUCTURED = "Structured"
    TELEGRAPHIC = "Telegraphic"
    UNSTRUCTURED = "Unstructured"


# --------------------------------------------------------------------------- Stage 1


class Demographics(BaseModel):
    model_config = ConfigDict(frozen=True)

    report_id: str = Field(pattern=REPORT_ID_PATTERN)
    age: int = Field(ge=18, le=100)
    sex: Literal["Female", "Male"]
    style: ReportingStyle

    @computed_field  # type: ignore[prop-decorator]
    @property
    def age_band(self) -> str:
        return age_band(self.age)


class PHIEntity(BaseModel):
    model_config = ConfigDict(frozen=True)

    phi_type: str
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _span_order(self) -> "PHIEntity":
        if self.end <= self.start:
            raise ValueError("PHI span end must be greater than start")
        return self


class DeidentificationResult(BaseModel):
    report_id: str
    phi_entities: list[PHIEntity] = Field(default_factory=list)
    redacted_text: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def phi_count(self) -> int:
        return len(self.phi_entities)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def safe_harbor_compliant_at_source(self) -> bool:
        """True when the *raw* report contained no detectable direct identifiers."""
        return not self.phi_entities


# --------------------------------------------------------------------------- Stage 2


class Section(BaseModel):
    name: SectionName
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    header: Optional[str] = None
    laterality: Optional[Laterality] = None
    method: Literal["header", "abbreviation", "semantic"]

    @property
    def label(self) -> str:
        return f"{self.name.value} ({self.laterality.value})" if self.laterality else self.name.value


# --------------------------------------------------------------------------- Stage 3/4


class Finding(BaseModel):
    """One normalised, grounded observation about one organ."""

    model_config = ConfigDict(validate_assignment=True)

    report_id: str = Field(pattern=REPORT_ID_PATTERN)
    finding_id: str
    organ: Organ
    finding: str = Field(min_length=2, description="Surface-faithful observation name")
    concept: str = Field(min_length=2, description="Normalised concept used for analytics")
    grade: Grade = Grade.NA
    size_mm: Optional[float] = Field(default=None, gt=0, le=300)
    laterality: Optional[Laterality] = None
    location: Optional[str] = None
    status: Assertion
    radlex_id: Optional[str] = None
    icd10_code: Optional[str] = None
    evidence_text: str = Field(min_length=2)
    span_start: int = Field(ge=0)
    span_end: int = Field(ge=0)
    section: str
    triage_tier: TriageTier = TriageTier.UNREMARKABLE
    extractor: str = "rules"
    flags: list[str] = Field(default_factory=list, description="Verification warnings, e.g. 'assertion_disagreement'")

    @field_validator("radlex_id")
    @classmethod
    def _radlex_format(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not RADLEX_PATTERN.match(value):
            raise ValueError(f"Malformed RadLex identifier: {value!r}")
        return value

    @field_validator("icd10_code")
    @classmethod
    def _icd_format(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not ICD10_PATTERN.match(value):
            raise ValueError(f"Malformed ICD-10-CM code: {value!r}")
        return value

    @model_validator(mode="after")
    def _clinical_consistency(self) -> "Finding":
        if self.grade is not Grade.NA and self.concept != "Steatosis":
            raise ValueError("Severity grades are only defined for hepatic steatosis")
        if self.icd10_code and self.status is not Assertion.AFFIRMED:
            # ICD-10-CM outpatient guidelines: negated or uncertain findings are not coded.
            raise ValueError("ICD-10-CM codes may only be attached to affirmed findings")
        if self.span_end < self.span_start:
            raise ValueError("Evidence span end precedes start")
        return self


class ReportConditions(BaseModel):
    """Report-level flags that drive population prevalence analytics."""

    steatosis: bool = False
    steatosis_grade: Grade = Grade.NA
    hepatomegaly: bool = False
    cholelithiasis: bool = False
    renal_cyst: bool = False
    renal_calculus: bool = False
    possible_renal_calculus: bool = False
    any_abnormality: bool = False


class StructuredReport(BaseModel):
    report_id: str = Field(pattern=REPORT_ID_PATTERN)
    demographics: Optional[Demographics] = None
    deidentification: DeidentificationResult
    sections: list[Section]
    findings: list[Finding]
    conditions: ReportConditions
    triage_tier: TriageTier
    triage_reasons: list[str] = Field(default_factory=list)
    processing_ms: float = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def triage_label(self) -> str:
        return self.triage_tier.label
