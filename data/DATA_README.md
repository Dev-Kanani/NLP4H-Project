# Vitals & Vectors: Abdominal USG Corpus v2 (100 reports)

| | reports | finding rows | Structured / Telegraphic / Unstructured | Female / Male |
|---|---|---|---|---|
| **v1 original** (US_SYNTH_001–050) | 50 | 614 | 30 / 10 / 10 | 25 / 25 |
| **v2 generated** (US_SYNTH_051–100) | 50 | 616 | 17 / 17 / 16 | 25 / 25 |
| **Total** | **100** | **1,230** | 47 / 27 / 26 | 50 / 50 |

The v1 files are byte-for-byte unchanged. v2 uses exactly the same file formats, so existing loaders work as-is.

## Files

| path | contents |
|---|---|
| `reports/US_SYNTH_001…100.txt` | raw report text |
| `reports_metadata.csv` | report_id, age, sex, style, num_findings, report_text |
| `ground_truth_findings.csv` | one row per finding: organ, finding, grade, size_mm, status, radlex_id, icd10_code, evidence_text |
| `ground_truth/US_SYNTH_xxx.json` | the same findings per report |
| `raw_batches/batch_01…10.txt` | report + metadata + gold JSON in one file per 10 reports |
| `phi_ground_truth.csv` | **new.** 27 identifier spans (names, dates, MRN/UHID, phones, facility) in reports 051, 062, 072, 084, 090 and 096, with character offsets, for scoring the de-identification stage. The v1 reports contain no identifiers. |
| `splits.csv` | **new.** dev = 001–075, **test = 076–100 (locked)** |
| `ANNOTATION_GUIDELINE.md` | v2.1 guideline, ontology table, and differences from v1 |
| `annotation_qc/` | both independent annotation passes, the adjudication log and agreement statistics |

## How v2 was made

1. **Design plan.** One line of clinical content per report fixed which findings exist (age 23–62, 25 F / 25 M,
   about one third of each style). It covers pathology absent from v1: gallbladder polyps, sludge,
   adenomyomatosis, post-cholecystectomy status, CBD dilatation, hepatic haemangioma and cyst, accessory spleen,
   renal calculi with hydronephrosis, angiomyolipoma, pancreatic calcification, prostatomegaly, fibroids,
   ovarian cysts, adenomyosis and physiological pelvic fluid. It also includes hedged findings ("?",
   "likely", "cannot be excluded").
2. **Writing.** Five independent Claude agents each wrote 10 reports freely in natural clinical English. They
   had no access to the extraction pipeline, its lexicon or the v1 reports. Each was told to give every report its
   own template and voice, to vary synonyms, negation and hedging styles, units, abbreviations ("B/L", "PCS",
   "CMD", "HDN", "s/o", "NAD") and length (90 to 2,650 characters), and to add occasional realistic typos. Each
   writer also labelled its own reports.
3. **Blind second annotation.** Five more agents labelled the text alone, without seeing the writers' labels.
4. **Adjudication.** 9 rows differed across 5 reports. All were resolved against the guideline, and the
   reasoning is in `annotation_qc/adjudication_log.csv`.
5. **Validation.** Every evidence span is an exact substring of its report. Vocabulary, grade, size and status
   rules were checked by script, as were PHI offsets and the consistency between metadata and gold.

**Agreement before adjudication:** entity-level F1 0.993. Assertion status, on 613 entities both passes
found, has Cohen's κ 0.997. Grades agree 26/26 and sizes 34/34. Both annotators are Claude instances, so this
measures **consistency between two LLM annotators, not human agreement.** Report it as such, and treat the
labels as silver standard until a human reviews at least the test split.

## Caveats

- This is still **synthetic data written by an LLM**. It varies far more than v1 but is cleaner than real PACS
  exports: no OCR noise, no mixed-language notes, no copy-paste artefacts.
- v1 and v2 differ in a few labelling conventions (guideline §8), mostly the "Unremarkable" rule. A score on
  all 100 reports mixes the two. Report v1 and v2 separately, or re-annotate v1 under v2.1.
- The ICD-10-CM and RadLex codes added in v2 should be verified before publication (guideline §7).
- **Keep 076–100 locked.** Tune rules or models on 001–075 only, and score on the test split once at the end.
  Reading test-set errors to fix rules turns it into a development set.
