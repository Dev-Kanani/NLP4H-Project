# Vitals & Vectors: Abdominal USG Report Structuring

**Course:** NLP for Healthcare (Sem 7), IIIT Hyderabad · **Team:** Dev Kanani, Kavya Bhalodi, Shivansh Santoki

This system turns free-text abdominal ultrasound reports across three distinct clinical styles (Structured, Telegraphic, and Unstructured) into validated, ontology-coded findings, an SQLite database, assistive follow-up triage tiers, workforce population health analytics, and an interactive Streamlit dashboard.

Two interchangeable extraction strategies sit behind a unified interface:

| extractor | how it reads text | runs on | role |
|---|---|---|---|
| `rules` | regex gazetteer + ConText negation | CPU, ~0.1 s for 75 reports | calibrated **baseline** |
| `llm` | open-weight instruction model (**Qwen 2.5 7B / Llama 3.1 8B / Mistral 7B**, ≤ 8B) with constrained JSON decoding + deterministic verifier | GPU via vLLM, or laptop via Ollama | **generalised** extractor |

### The Generalization Gap
The rule-based extractor was calibrated based on the initial 50 reports, scoring **F1 0.994** on that cohort. However, when evaluated on subsequent reports with independent clinical phrasing and shorthand, its performance drops to **0.725** (and 0.68 on the locked test split). This performance drop illustrates the brittleness of static regular expressions. The LLM extractor holds no hard-coded keyword lists; its vocabulary, codes, and extraction policy derive entirely from the ontology, and every model proposal is deterministically verified against the source text before storage.

```bash
pip install -r requirements.txt
python run_pipeline.py --split dev                     # rule baseline (CPU)
python -m pytest -q                                    # 52 tests, incl. mock-model LLM tests
streamlit run dashboard/app.py
```

---

## 1. System Architecture

```
report.txt ──► PHIScanner (de-identify) ──► AnatomicalSectionizer (display / style)
                         │ de-identified text only ever reaches the model
                         ▼
          ┌──────────────┴───────────────┐
   RuleBasedExtractor             LLMExtractor
   (lexicon regex +               PromptBuilder: guideline rules + vocabulary from ontology
    ConText)                        + 3 dev-split examples (never test, never the same report)
          │                       Backend: Ollama │ OpenAI-compatible (vLLM, llama.cpp) │ transformers │ replay
          │                         + JSON-schema constrained decoding, parse-retry, response cache
          │                       FindingVerifier:
          │                         • vocabulary normalisation (exact → alias → fuzzy → organ correction)
          │                         • evidence grounding (exact → normalised → fuzzy) - else REJECT (hallucination)
          │                         • attribute sanity (grade only steatosis; size must appear in text)
          │                         • ConText cross-check → flag disagreements (audit flag)
          └──────────────┬───────────────┘
                         ▼
   ontology.json ──► codes (RadLex / ICD-10-CM, ICD only if Affirmed) ──► triage tiers ──► Pydantic ──► SQLite
                         ▼
   Evaluator (split-aware, PHI, reliability) ──► outputs/<run>/ ──► dashboard / --compare
```

| Path | Responsibility |
|---|---|
| `usg_nlp/resources/ontology.json` | **Single source of clinical knowledge**: organs, 40 finding concepts with definitions, aliases, RadLex/ICD codes, and triage tiers |
| `usg_nlp/ontology.py` | Loader, vocabulary normalisation, code and triage lookup |
| `usg_nlp/llm/backends.py` | Ollama, OpenAI-compatible (vLLM), transformers, replay and mock backends; response caching |
| `usg_nlp/llm/prompts.py` | Prompt construction from guidelines + ontology; dev-only few-shot selection; JSON schema |
| `usg_nlp/extractors.py` | `BaseExtractor`, `RuleBasedExtractor`, `LLMExtractor` (retries, cache, diagnostics) |
| `usg_nlp/verification.py` | Evidence grounding, normalisation, attribute checks, ConText cross-check |
| `usg_nlp/extractor.py`, `lexicon.py`, `sectionizer.py`, `context.py` | Rule baseline engine (regex gazetteer + ConText algorithm) |
| `usg_nlp/deid.py`, `ingestion.py`, `schemas.py`, `triage.py`, `database.py`, `analytics.py` | Shared pipeline stages: de-identification, SQLite persistence, and cohort analytics |
| `usg_nlp/evaluation.py` | Benchmark suite with cluster bootstrapping; logs test-split runs |
| `run_pipeline.py` | Unified CLI (`--extractor`, `--backend`, `--model`, `--split`, `--compare`) |
| `scripts/ada_vllm.sbatch`, `scripts/compare_models.sh` | Cluster (vLLM) and local execution scripts |

---

## 2. Interactive Streamlit Dashboard

Launch the interactive clinical interface:

```bash
streamlit run dashboard/app.py
```

The dashboard includes 4 operational views:

### Figure 1: Interactive Report Structurer
*Live clinical inspection showing automatic organ sectionization, PHI redaction, and color-coded evidence spans overlaid directly on the de-identified report.*
![Figure 1: Interactive Report Structurer](page1.png)

### Figure 2: Structured Findings Table with Clinical Codes & Triage
*Granular entity extraction mapping each sonographic finding to standard RadLex ontology IDs, ICD-10-CM diagnostic codes, and actionable follow-up triage tiers.*
![Figure 2: Structured Findings Table](d5.png)

### Figure 3: Workforce Population Health Analytics
*Epidemiological cohort dashboard showing demographic distributions, condition prevalence with 95% Wilson score confidence intervals, and follow-up tier distributions.*
![Figure 3: Workforce Population Health Analytics](page2.png)

### Figure 4: Evaluation Benchmark Suite
*Comprehensive evaluation suite presenting micro Precision/Recall/F1 with 95% cluster-bootstrap confidence intervals, style-stratified performance, assertion confusion, and error diagnostics.*
![Figure 4: Evaluation Suite](page3.png)

---

## 3. Running the LLM Extractor

The model never encounters protected health information: de-identification executes first. Every response is cached in `outputs/llm_cache/<model>.jsonl`, allowing runs to be **replayed exactly without the model** (for grading, the dashboard, or CI).

**Workstation / Laptop (Ollama):**
```bash
ollama pull qwen2.5:7b-instruct
python run_pipeline.py --extractor llm --backend ollama --model qwen2.5:7b-instruct --split dev
```

**GPU Cluster via Slurm (vLLM on Ada):**
```bash
sbatch scripts/ada_vllm.sbatch                                        # Qwen2.5-7B-Instruct
MODEL=mistralai/Mistral-7B-Instruct-v0.3 sbatch scripts/ada_vllm.sbatch
MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=hf_... sbatch scripts/ada_vllm.sbatch
```

**Any OpenAI-Compatible Server (vLLM / llama.cpp):**
```bash
python run_pipeline.py --extractor llm --backend openai --base-url http://HOST:8000/v1 --model <name> --split dev
```

**Replay a Recorded Run & Compare:**
```bash
python run_pipeline.py --extractor llm --backend replay --model qwen2.5:7b-instruct --split dev
python run_pipeline.py --compare
```

---

## 4. Dataset and Evaluation Protocol

The dataset comprises **100 abdominal ultrasound reports** in total, spanning three distinct clinical writing styles (47 Structured, 27 Telegraphic, 26 Unstructured).

| Split | Reports | Purpose |
|---|---|---|
| `dev` | 001–075 | Rule calibration (001–050), prompt design, few-shot pool, error analysis |
| `test` | **076–100** | **Locked test set**: evaluated once for final comparison |

The benchmark reports micro P/R/F1 with 95% cluster-bootstrap CIs, broken down by reporting style and by calibration vs unseen phrasing. It also measures assertion accuracy and Cohen's κ, numeric size MAE, steatosis grade accuracy, prevalence estimation error, follow-up tier agreement, and PHI de-identification recall. For model runs, it tracks reliability metrics: proposed entities, hallucination rate (ungrounded evidence), out-of-vocabulary rate, and parse failures.

### Current Results (Development Split)

| Run | Overall F1 | Calibration Reports (001–050) | Unseen Dev Reports (051–075) | Notes |
|---|---|---|---|---|
| `rules` | 0.916 [0.885–0.946] | 0.994 | **0.725** | Rules calibrated on 001–050; drops on unseen wording |
| mock oracle (gold answers) | 1.000 | 1.000 | 1.000 | Plumbing check proving zero pipeline information loss |
| Qwen / Llama / Mistral | *ready for cluster* | — | — | Configured in `scripts/ada_vllm.sbatch` |

The rule baseline scores 0.68 on the locked test split. That figure was measured once for the generalization-gap analysis before designing the LLM architecture.

**Plumbing Verification:** When the extractor returns the gold annotations, every downstream stage (prompting → parsing → grounding → vocabulary normalization → coding → database → evaluation) scores F1 = 1.000. Any performance drop in real model runs is thus attributable to model capability, not pipeline loss.

---

## 5. Current Implementation Status & Notes

- **Completed & Verified in Codebase:**
  - **Single Unified Pipeline:** End-to-end processing: raw text ingestion → PHI de-identification → organ sectionization → extraction → deterministic verification → database storage → evaluation.
  - **Rule-Based Baseline:** Fully implemented, verified, and benchmarked on the 75-report development split (0.916 overall F1). Rules calibrated on the initial 50 reports (0.994 F1) demonstrate a clear drop on unseen phrasing (0.725 F1).
  - **Corpus & Ground Truth:** 100 abdominal ultrasound reports annotated under clinical guidelines with high dual-annotator agreement (Cohen's $\kappa = 0.997$).
  - **Deterministic Verification Layer:** Strict evidence grounding to source text (rejects hallucinations), ontology vocabulary mapping (rejects out-of-vocabulary proposals), attribute sanity checks (grade and size validation), and ConText cross-checks.
  - **Interactive Streamlit Dashboard:** 4 operational views (Report Structurer, Population Health Analytics, Evaluation Suite, System Architecture).
  - **Test Suite:** 52/52 pytest tests passing cleanly across all modules and mock backends.
- **Next Milestone (Underway):**
  - Executing open-weight instruction models (Qwen 2.5 7B, Llama 3.1 8B, Mistral 7B) on the IIIT Ada GPU cluster using `scripts/ada_vllm.sbatch`.
  - Final blind evaluation on the locked test split (`076-100`).
