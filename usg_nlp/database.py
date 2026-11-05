"""Stage 4b - relational persistence in an indexed SQLite database.

Schema (3NF, one row per real-world entity)::

    patients(report_id PK, age, sex, style, age_band)
    reports(report_id PK/FK, raw_text, deid_text, phi_count, style_detected,
            triage_tier, triage_label, triage_reasons, processing_ms, ...flags)
    sections(section_id PK, report_id FK, name, laterality, method, header, text)
    findings(finding_id PK, report_id FK, organ, finding, concept, grade, size_mm,
             laterality, location, status, radlex_id, icd10_code, evidence_text,
             span_start, span_end, section, triage_tier, extractor, flags)
    phi_audit(id PK, report_id FK, phi_type, text, start, end)
    run_metadata(key PK, value)

Views::

    v_patient_conditions     one row per patient with demographic + condition flags
    v_condition_prevalence   cohort prevalence of each target condition
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .schemas import StructuredReport

DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS patients (
    report_id   TEXT PRIMARY KEY,
    age         INTEGER NOT NULL CHECK (age BETWEEN 18 AND 100),
    sex         TEXT    NOT NULL CHECK (sex IN ('Female','Male')),
    style       TEXT    NOT NULL,
    age_band    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    report_id               TEXT PRIMARY KEY REFERENCES patients(report_id) ON DELETE CASCADE,
    raw_text                TEXT NOT NULL,
    deid_text               TEXT NOT NULL,
    phi_count               INTEGER NOT NULL,
    style_detected          TEXT,
    triage_tier             INTEGER NOT NULL CHECK (triage_tier IN (1,2,3)),
    triage_label            TEXT NOT NULL,
    triage_reasons          TEXT NOT NULL,
    steatosis               INTEGER NOT NULL,
    steatosis_grade         TEXT NOT NULL,
    hepatomegaly            INTEGER NOT NULL,
    cholelithiasis          INTEGER NOT NULL,
    renal_cyst              INTEGER NOT NULL,
    renal_calculus          INTEGER NOT NULL,
    possible_renal_calculus INTEGER NOT NULL,
    any_abnormality         INTEGER NOT NULL,
    processing_ms           REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sections (
    section_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id   TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    laterality  TEXT,
    method      TEXT NOT NULL,
    header      TEXT,
    text        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS findings (
    finding_id    TEXT PRIMARY KEY,
    report_id     TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    organ         TEXT NOT NULL,
    finding       TEXT NOT NULL,
    concept       TEXT NOT NULL,
    grade         TEXT NOT NULL,
    size_mm       REAL,
    laterality    TEXT,
    location      TEXT,
    status        TEXT NOT NULL CHECK (status IN ('Affirmed','Negated','Possible')),
    radlex_id     TEXT,
    icd10_code    TEXT,
    evidence_text TEXT NOT NULL,
    span_start    INTEGER NOT NULL,
    span_end      INTEGER NOT NULL,
    section       TEXT NOT NULL,
    triage_tier   INTEGER NOT NULL,
    extractor     TEXT NOT NULL,
    flags         TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS phi_audit (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id  TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    phi_type   TEXT NOT NULL,
    text       TEXT NOT NULL,
    start      INTEGER NOT NULL,
    "end"      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS run_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE INDEX IF NOT EXISTS ix_patients_band_sex   ON patients(age_band, sex);
CREATE INDEX IF NOT EXISTS ix_findings_report     ON findings(report_id);
CREATE INDEX IF NOT EXISTS ix_findings_concept    ON findings(concept, status);
CREATE INDEX IF NOT EXISTS ix_findings_organ      ON findings(organ, concept);
CREATE INDEX IF NOT EXISTS ix_findings_icd        ON findings(icd10_code);
CREATE INDEX IF NOT EXISTS ix_reports_tier        ON reports(triage_tier);
CREATE INDEX IF NOT EXISTS ix_sections_report     ON sections(report_id);
CREATE INDEX IF NOT EXISTS ix_findings_extractor   ON findings(extractor);

CREATE VIEW IF NOT EXISTS v_patient_conditions AS
SELECT p.report_id, p.age, p.sex, p.style, p.age_band,
       r.triage_tier, r.triage_label, r.steatosis, r.steatosis_grade, r.hepatomegaly,
       r.cholelithiasis, r.renal_cyst, r.renal_calculus, r.possible_renal_calculus, r.any_abnormality
FROM patients p JOIN reports r USING (report_id);

CREATE VIEW IF NOT EXISTS v_condition_prevalence AS
SELECT 'Hepatic steatosis' AS condition, AVG(steatosis) AS prevalence, SUM(steatosis) AS n, COUNT(*) AS cohort FROM reports
UNION ALL SELECT 'Hepatomegaly', AVG(hepatomegaly), SUM(hepatomegaly), COUNT(*) FROM reports
UNION ALL SELECT 'Cholelithiasis', AVG(cholelithiasis), SUM(cholelithiasis), COUNT(*) FROM reports
UNION ALL SELECT 'Renal cortical cyst', AVG(renal_cyst), SUM(renal_cyst), COUNT(*) FROM reports
UNION ALL SELECT 'Possible renal calculus', AVG(possible_renal_calculus), SUM(possible_renal_calculus), COUNT(*) FROM reports
UNION ALL SELECT 'Any abnormality', AVG(any_abnormality), SUM(any_abnormality), COUNT(*) FROM reports;
"""


class ClinicalDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def reset(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()
        with closing(self.connect()) as conn:
            conn.executescript(DDL)
            conn.commit()

    def write(self, reports: Iterable[StructuredReport], raw_texts: dict[str, str], metadata: dict[str, str]) -> None:
        self.reset()
        with closing(self.connect()) as conn, conn:
            for rep in reports:
                demo = rep.demographics
                if demo is None:
                    continue  # patients row is mandatory for referential integrity
                conn.execute(
                    "INSERT INTO patients VALUES (?,?,?,?,?)",
                    (demo.report_id, demo.age, demo.sex, demo.style.value, demo.age_band),
                )
                c = rep.conditions
                conn.execute(
                    "INSERT INTO reports VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        rep.report_id, raw_texts[rep.report_id], rep.deidentification.redacted_text,
                        rep.deidentification.phi_count, metadata.get(f"style_detected:{rep.report_id}"),
                        int(rep.triage_tier), rep.triage_label, json.dumps(rep.triage_reasons),
                        int(c.steatosis), c.steatosis_grade.value, int(c.hepatomegaly), int(c.cholelithiasis),
                        int(c.renal_cyst), int(c.renal_calculus), int(c.possible_renal_calculus),
                        int(c.any_abnormality), rep.processing_ms,
                    ),
                )
                conn.executemany(
                    "INSERT INTO sections(report_id,name,laterality,method,header,text) VALUES (?,?,?,?,?,?)",
                    [
                        (rep.report_id, s.name.value, s.laterality.value if s.laterality else None, s.method, s.header, s.text)
                        for s in rep.sections
                    ],
                )
                conn.executemany(
                    "INSERT INTO findings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            f.finding_id, f.report_id, f.organ.value, f.finding, f.concept, f.grade.value, f.size_mm,
                            f.laterality.value if f.laterality else None, f.location, f.status.value, f.radlex_id,
                            f.icd10_code, f.evidence_text, f.span_start, f.span_end, f.section, int(f.triage_tier),
                            f.extractor, json.dumps(f.flags),
                        )
                        for f in rep.findings
                    ],
                )
                conn.executemany(
                    'INSERT INTO phi_audit(report_id,phi_type,text,start,"end") VALUES (?,?,?,?,?)',
                    [(rep.report_id, e.phi_type, e.text, e.start, e.end) for e in rep.deidentification.phi_entities],
                )
            meta = {**{k: v for k, v in metadata.items() if not k.startswith("style_detected:")},
                    "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            conn.executemany("INSERT INTO run_metadata VALUES (?,?)", list(meta.items()))

    # ------------------------------------------------------------------ read helpers (dashboard, tests)
    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with closing(self.connect()) as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def table(self, name: str) -> pd.DataFrame:
        if name not in {"patients", "reports", "sections", "findings", "phi_audit", "run_metadata",
                        "v_patient_conditions", "v_condition_prevalence"}:
            raise ValueError(f"Unknown table {name!r}")
        return self.query(f"SELECT * FROM {name}")
