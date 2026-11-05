"""Prompt construction for the LLM extractor.

The prompt is assembled from data, not hand-written per finding:

* the **instructions** restate the annotation guideline (what counts as a
  finding, status definitions, attribute rules) - the same specification the
  human/LLM annotators followed;
* the **vocabulary** table is generated from ``ontology.json`` (finding names +
  one-line definitions per organ);
* **few-shot examples** are gold-annotated reports drawn *only* from the
  development split, one per reporting style, never the report being
  processed and never a test-split report;
* the **output schema** is derived from the ontology, so grammar-constrained
  decoding can only emit valid organ / finding / status values.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..ontology import Ontology, load_ontology

LOGGER = logging.getLogger(__name__)
PROMPT_VERSION = "v2.1-1"
LLM_ORGANS = ["Liver", "Gallbladder", "Pancreas", "Spleen", "Kidneys", "Pelvis", "Global"]

INSTRUCTIONS = """You are a clinical information-extraction system for abdominal and pelvic ultrasound reports \
from preventive health check-ups. Read the report and list every finding it states, following these rules exactly.

WHAT TO EXTRACT
1. Extract a finding whenever the report explicitly names it (or an unambiguous synonym/abbreviation), whether \
it is present, absent or uncertain. "No calculus or hydronephrosis" gives two findings.
2. Do NOT extract plain normal attribute descriptions that do not name a finding ("normal in size", \
"normal echotexture", "CMD maintained", "CBD 4 mm", organ measurements). "No hepatomegaly" IS a finding (Negated).
3. Unremarkable: when an organ is declared normal as a whole ("unremarkable", "NAD", "normal", "appears normal", \
"WNL" as the whole statement about it), output finding "Unremarkable" for that organ. Never output Unremarkable for \
an organ that also has an Affirmed or Possible abnormality. Separate statements give separate rows; one combined \
statement gives one row per organ it covers. "Rest of abdomen NAD" / "otherwise normal" gives nothing.
4. Unremarkable Study: if the report concludes the whole examination is normal (without "otherwise", "no other", \
"rest"), output one finding "Unremarkable Study" with organ "Global".
5. Impression/conclusion sections usually repeat the body: do not output repeats. Only add an impression statement \
that names a finding the body never mentioned.
6. One row per mention per sub-section: separate RIGHT and LEFT kidney lines that both rule out hydronephrosis give \
two rows. Separately described lesions with their own sizes give separate rows; a group ("multiple calculi, \
largest 7 mm") gives one row with the largest size.
7. Do not extract age-appropriate or technical states that have no vocabulary name (atrophic postmenopausal uterus, \
ovaries not visualised, contracted gallbladder, partial-organ statements), advice, or patient identifiers.

STATUS
- Affirmed: stated present, including "consistent with", "suggestive of", "s/o", "in keeping with", "features of", \
"physiological".
- Negated: "no", "not seen", "absent", "without", "nil", "neither ... nor", "ruled out", "negative" (Murphy sign \
negative), "not dilated", "not thickened".
- Possible: "?", "possible", "probable", "likely", "suspicious for", "may represent", "could be", "cannot be \
excluded/ruled out", "query", "equivocal". If a lesion is definitely seen and only its nature is uncertain, it is \
Affirmed; if its existence is uncertain ("? tiny calculus", "apparent thickening, likely physiological"), Possible.

ATTRIBUTES
- grade: only for Steatosis: "Grade 1"/"Grade 2"/"Grade 3" from an explicit grade or mild/moderate/severe; else null.
- size_mm: only for stones, cysts, polyps, lesions, masses, nodules, fibroids: largest dimension converted to \
millimetres (1.2 x 0.9 cm -> 12). Never organ sizes, duct calibres, wall thickness or volumes. null when Negated.
- laterality: "Left", "Right" or "Bilateral" when the report states the side, else null.
- evidence: copy the shortest exact phrase from the report that shows the finding and its status cue. Copy it \
character for character; do not paraphrase.

Use only the organ and finding names in the vocabulary below. Output a single JSON object of the form \
{"findings": [...]} and nothing else."""


@dataclass(frozen=True)
class FewShotExample:
    report_id: str
    style: str
    text: str
    answer: dict


class PromptBuilder:
    def __init__(self, ontology: Optional[Ontology] = None, examples: Optional[list[FewShotExample]] = None) -> None:
        self.ontology = ontology or load_ontology()
        self.examples = examples or []

    # ------------------------------------------------------------------ schema
    def output_schema(self) -> dict:
        finding_names = [f for f in self.ontology.all_findings() if f]
        item = {
            "type": "object",
            "properties": {
                "organ": {"type": "string", "enum": LLM_ORGANS},
                "finding": {"type": "string", "enum": finding_names},
                "status": {"type": "string", "enum": ["Affirmed", "Negated", "Possible"]},
                "grade": {"type": ["string", "null"], "enum": ["Grade 1", "Grade 2", "Grade 3", None]},
                "size_mm": {"type": ["number", "null"]},
                "laterality": {"type": ["string", "null"], "enum": ["Left", "Right", "Bilateral", None]},
                "evidence": {"type": "string"},
            },
            "required": ["organ", "finding", "status", "grade", "size_mm", "laterality", "evidence"],
            "additionalProperties": False,
        }
        return {"type": "object", "properties": {"findings": {"type": "array", "items": item}},
                "required": ["findings"], "additionalProperties": False}

    # ------------------------------------------------------------------ text blocks
    def vocabulary_block(self) -> str:
        lines = ["VOCABULARY (organ -> finding: definition)"]
        for organ in LLM_ORGANS:
            entries = [e for e in self.ontology.entries if organ in e.organs and not e.legacy]
            if not entries:
                continue
            lines.append(f"{organ}:")
            for e in entries:
                lines.append(f"  - {e.finding}: {e.definition}")
        return "\n".join(lines)

    def system_prompt(self) -> str:
        return f"{INSTRUCTIONS}\n\n{self.vocabulary_block()}"

    def build(self, report_text: str, exclude_report_id: Optional[str] = None) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt()}]
        for ex in self.examples:
            if ex.report_id == exclude_report_id:
                continue
            messages.append({"role": "user", "content": f"REPORT:\n{ex.text}"})
            messages.append({"role": "assistant", "content": json.dumps(ex.answer, ensure_ascii=False)})
        messages.append({"role": "user", "content": f"REPORT:\n{report_text}"})
        return messages


# --------------------------------------------------------------------------- few-shot selection


def gold_to_answer(rows: list[dict], ontology: Ontology) -> dict:
    """Convert gold CSV rows into the exact JSON the model is asked to produce."""
    out = []
    for r in rows:
        finding = r["finding"]
        organ = "Global" if ontology.study_level(finding) else r["organ"]
        size = r.get("size_mm")
        out.append({
            "organ": organ, "finding": finding, "status": r["status"],
            "grade": r["grade"] if r.get("grade") not in (None, "", "N/A") else None,
            "size_mm": float(size) if size not in (None, "", "nan") else None,
            "laterality": None,
            "evidence": r["evidence_text"],
        })
    return {"findings": out}


def select_examples(data_dir: Path, n: int = 3, source_prefix_min: str = "US_SYNTH_051",
                    ontology: Optional[Ontology] = None, max_chars: int = 1500) -> list[FewShotExample]:
    """Pick ``n`` dev-split examples, one per style in turn, preferring v2-guideline reports.

    Raises if the splits file marks a chosen report as test, so test data can never
    leak into the prompt.
    """
    if n <= 0:
        return []
    ontology = ontology or load_ontology()
    data_dir = Path(data_dir)
    splits = {}
    split_csv = data_dir / "splits.csv"
    if split_csv.exists():
        splits = {r["report_id"]: r["split"] for r in csv.DictReader(split_csv.open(encoding="utf-8"))}
    else:
        LOGGER.warning("No splits.csv: few-shot examples cannot be guaranteed to be outside a test set")
    meta = {r["report_id"]: r for r in csv.DictReader((data_dir / "reports_metadata.csv").open(encoding="utf-8"))}
    gold: dict[str, list[dict]] = {}
    for r in csv.DictReader((data_dir / "ground_truth_findings.csv").open(encoding="utf-8")):
        gold.setdefault(r["report_id"], []).append(r)

    candidates = []
    for rid, m in meta.items():
        if rid < source_prefix_min or splits.get(rid, "dev") != "dev" or rid not in gold:
            continue
        text = (data_dir / "reports" / f"{rid}.txt").read_text(encoding="utf-8").strip()
        if len(text) > max_chars:
            continue
        variety = len({(g["organ"], g["finding"], g["status"]) for g in gold[rid]})
        candidates.append((m["style"], -variety, len(text), rid, text))
    by_style: dict[str, list] = {}
    for c in sorted(candidates):
        by_style.setdefault(c[0], []).append(c)
    chosen, styles = [], ["Telegraphic", "Unstructured", "Structured"]
    i = 0
    while len(chosen) < n and any(by_style.values()):
        st = styles[i % len(styles)]
        if by_style.get(st):
            style, _, _, rid, text = by_style[st].pop(0)
            assert splits.get(rid, "dev") == "dev", f"{rid} is not a dev report"
            chosen.append(FewShotExample(rid, style, text, gold_to_answer(gold[rid], ontology)))
        i += 1
    return chosen
