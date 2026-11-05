"""Stage 5 - benchmark against the gold-standard annotations.

Matching protocol
-----------------
An *entity* is ``(report_id, organ, finding, status)``.  Matching is
**multiset** intersection per report (telegraphic ``RK:``/``LK:`` lines can
legitimately repeat a negation).

* ``concept`` mode (primary): finding strings are mapped through the ontology
  to their concept (laterality suffixes and v1 surface variants absorbed;
  whole-study normality aligned to organ ``Global``).
* ``strict`` mode: the exact finding string must match (lower bound).

Only reports processed by the run are scored, so a ``--split dev`` run is
scored on dev gold only.  Results are stratified by reporting style and by
corpus source (v1 original vs v2 generated, whose labelling conventions differ
- see ``data/ANNOTATION_GUIDELINE.md`` section 8).

Statistics: micro P/R/F1, 95% percentile **cluster bootstrap over reports**;
assertion accuracy / macro-F1 / kappa on concept-aligned pairs; size MAE;
steatosis grade accuracy; prevalence error; report follow-up tier agreement;
PHI recall/false positives against ``phi_ground_truth.csv``; and, for model
extractors, proposal-level reliability (hallucination = ungrounded evidence,
out-of-vocabulary, parse failures).

Every evaluation that touches a ``test``-split report is appended to
``outputs/test_set_evaluations.log`` so the number of looks at the test set is
auditable.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .ontology import load_ontology
from .pipeline import PipelineRun

STATUSES = ("Affirmed", "Negated", "Possible")
STYLES = ("Structured", "Telegraphic", "Unstructured")


def normalise_concept(finding: str, organ: str = "") -> str:
    return load_ontology().concept_of(organ, finding)


def normalise_organ(organ: str, concept: str) -> str:
    return "Global" if concept == "Unremarkable Study" else organ


def _clean_code(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value = str(value).strip()
    return None if value in {"", "N/A", "NA", "nan", "None"} else value


# --------------------------------------------------------------------------- data prep


def prepare_gold(gold_csv) -> pd.DataFrame:
    gold = pd.read_csv(gold_csv, keep_default_na=False, na_values={"size_mm": [""]})
    gold["concept"] = [normalise_concept(f, o) for f, o in zip(gold["finding"], gold["organ"])]
    gold["organ_norm"] = [normalise_organ(o, c) for o, c in zip(gold["organ"], gold["concept"])]
    gold["finding_strict"] = gold["finding"].astype(str)
    gold["radlex_id"] = gold["radlex_id"].map(_clean_code)
    gold["icd10_code"] = gold["icd10_code"].map(_clean_code)
    gold["grade"] = gold["grade"].replace("", "N/A")
    gold["size_mm"] = pd.to_numeric(gold["size_mm"], errors="coerce")
    return gold


def prepare_predictions(run: PipelineRun) -> pd.DataFrame:
    pred = run.findings_frame()
    if pred.empty:
        pred = pd.DataFrame(columns=["report_id", "organ", "finding", "concept", "status", "grade", "size_mm",
                                     "radlex_id", "icd10_code", "evidence_text", "flags"])
    pred["concept"] = [normalise_concept(f, o) for f, o in zip(pred["finding"], pred["organ"])]
    pred["organ_norm"] = [normalise_organ(o, c) for o, c in zip(pred["organ"], pred["concept"])]
    pred["finding_strict"] = pred["finding"].astype(str)
    return pred


# --------------------------------------------------------------------------- core metrics


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@dataclass
class CountTable:
    report_ids: list[str]
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    errors: list[dict] = field(default_factory=list)


def multiset_counts(gold: pd.DataFrame, pred: pd.DataFrame, report_ids: list[str],
                    key: Callable[[pd.Series], tuple]) -> CountTable:
    g_groups = {rid: Counter(key(row) for _, row in grp.iterrows()) for rid, grp in gold.groupby("report_id")}
    p_groups = {rid: Counter(key(row) for _, row in grp.iterrows()) for rid, grp in pred.groupby("report_id")}
    tp, fp, fn, errors = [], [], [], []
    for rid in report_ids:
        g, p = g_groups.get(rid, Counter()), p_groups.get(rid, Counter())
        inter = g & p
        tp.append(sum(inter.values()))
        fp.append(sum((p - g).values()))
        fn.append(sum((g - p).values()))
        for k, n in (p - g).items():
            errors.extend({"report_id": rid, "error": "FP", "key": " | ".join(map(str, k))} for _ in range(n))
        for k, n in (g - p).items():
            errors.extend({"report_id": rid, "error": "FN", "key": " | ".join(map(str, k))} for _ in range(n))
    return CountTable(report_ids, np.array(tp, float), np.array(fp, float), np.array(fn, float), errors)


def bootstrap_prf(table: CountTable, n_resamples: int, seed: int, level: float) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    n = len(table.report_ids)
    idx = rng.integers(0, n, size=(n_resamples, n))
    tp, fp, fn = table.tp[idx].sum(1), table.fp[idx].sum(1), table.fn[idx].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {name: [float(np.percentile(arr, lo)), float(np.percentile(arr, hi))]
            for name, arr in (("precision", p), ("recall", r), ("f1", f))}


def summarise(table: CountTable, cfg: PipelineConfig) -> dict:
    tp, fp, fn = table.tp.sum(), table.fp.sum(), table.fn.sum()
    p, r, f = prf(tp, fp, fn)
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": p, "recall": r, "f1": f,
        "ci95": bootstrap_prf(table, cfg.bootstrap_resamples, cfg.bootstrap_seed, cfg.confidence_level),
    }


def align_by_concept(gold: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Pair gold and predicted entities sharing (report, organ, concept), preferring same-status pairs."""
    rows = []
    g_idx = gold.groupby(["report_id", "organ_norm", "concept"])
    p_idx = {k: grp for k, grp in pred.groupby(["report_id", "organ_norm", "concept"])}
    for k, g_grp in g_idx:
        p_grp = p_idx.get(k)
        if p_grp is None:
            continue
        g_items = g_grp.to_dict("records")
        p_items = p_grp.to_dict("records")
        used_p: set[int] = set()
        pending: list[dict] = []
        for g in g_items:  # pass 1: same-status pairs
            j = next((j for j, p in enumerate(p_items) if j not in used_p and p["status"] == g["status"]), None)
            if j is None:
                pending.append(g)
            else:
                used_p.add(j)
                rows.append((g, p_items[j]))
        for g in pending:  # pass 2: residual pairs -> assertion disagreements
            j = next((j for j in range(len(p_items)) if j not in used_p), None)
            if j is not None:
                used_p.add(j)
                rows.append((g, p_items[j]))
    return pd.DataFrame(
        [{
            "report_id": g["report_id"], "organ": g["organ_norm"], "concept": g["concept"],
            "gold_status": g["status"], "pred_status": p["status"],
            "gold_grade": g["grade"], "pred_grade": p["grade"],
            "gold_size": g["size_mm"], "pred_size": p["size_mm"],
            "gold_radlex": g["radlex_id"], "pred_radlex": _clean_code(p["radlex_id"]),
            "gold_icd": g["icd10_code"], "pred_icd": _clean_code(p["icd10_code"]),
            "gold_evidence": g["evidence_text"], "pred_evidence": p["evidence_text"],
        } for g, p in rows]
    )


def cohen_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    if not a:
        return float("nan")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def macro_f1(gold: list[str], pred: list[str], labels=STATUSES) -> float:
    scores = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        if tp + fp + fn:
            scores.append(prf(tp, fp, fn)[2])
    return float(np.mean(scores)) if scores else float("nan")


def _report_flags(df: pd.DataFrame, report_ids: list[str]) -> pd.DataFrame:
    def has(mask):
        return set(df.loc[mask, "report_id"])

    aff = df["status"] == "Affirmed"
    conds = {
        "Hepatic steatosis": has((df["concept"] == "Steatosis") & aff),
        "Hepatomegaly": has((df["concept"] == "Hepatomegaly") & aff),
        "Cholelithiasis": has((df["concept"] == "Calculus") & (df["organ_norm"] == "Gallbladder") & aff),
        "Gallbladder polyp": has((df["concept"] == "Polyp") & aff),
        "Renal calculus": has((df["concept"] == "Calculus") & (df["organ_norm"] == "Kidneys") & aff),
        "Renal cortical cyst": has((df["concept"] == "Cortical Cyst") & aff),
        "Possible renal calculus": has((df["concept"] == "Calculus") & (df["organ_norm"] == "Kidneys")
                                       & (df["status"] == "Possible")),
        "Prostatomegaly": has((df["concept"] == "Prostatomegaly") & aff),
        "Uterine fibroid": has((df["concept"] == "Uterine Fibroid") & aff),
    }
    return pd.DataFrame({c: [rid in s for rid in report_ids] for c, s in conds.items()}, index=report_ids)


def _tier(df: pd.DataFrame, report_ids: list[str]) -> dict[str, int]:
    onto = load_ontology()
    tiers = {rid: 1 for rid in report_ids}
    for row in df.itertuples():
        rule = onto.triage(row.organ_norm if row.organ_norm != "Global" else "Global",
                           row.finding_strict.split(" - ")[0], row.status, row.grade)
        if rule and row.report_id in tiers:
            tiers[row.report_id] = max(tiers[row.report_id], rule[0])
    return tiers


def _read_csv_dicts(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def evaluate_phi(run: PipelineRun, phi_gold_csv) -> dict:
    """Span coverage of gold identifiers by the de-identification scanner (FACILITY rows are optional)."""
    try:
        gold = [r for r in _read_csv_dicts(phi_gold_csv)]
    except FileNotFoundError:
        return {"available": False}
    detected = {r.report_id: r.deidentification.phi_entities for r in run.reports}
    gold = [g for g in gold if g["report_id"] in detected]
    required = [g for g in gold if g["phi_type"] != "FACILITY"]
    covered = 0
    missed = []
    for g in required:
        s, e = int(g["start"]), int(g["end"])
        hits = [h for h in detected[g["report_id"]] if h.start < e and h.end > s]
        if hits and min(h.start for h in hits) <= s and max(h.end for h in hits) >= e:
            covered += 1
        else:
            missed.append({"report_id": g["report_id"], "type": g["phi_type"], "text": g["text"]})
    false_pos = []
    for rid, ents in detected.items():
        spans = [(int(g["start"]), int(g["end"])) for g in gold if g["report_id"] == rid]
        for h in ents:
            if not any(h.start < e and h.end > s for s, e in spans):
                false_pos.append({"report_id": rid, "type": h.phi_type, "text": h.text})
    return {"available": True, "gold_identifiers": len(required), "covered": covered,
            "recall": covered / len(required) if required else float("nan"),
            "false_positives": len(false_pos), "missed": missed, "false_positive_examples": false_pos[:20]}


# --------------------------------------------------------------------------- orchestrator


@dataclass
class EvaluationResult:
    summary: dict
    by_style: pd.DataFrame
    by_concept: pd.DataFrame
    aligned: pd.DataFrame
    errors: pd.DataFrame
    prevalence: pd.DataFrame
    by_source: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_json(self) -> str:
        return json.dumps(self.summary, indent=2, default=float)


class Evaluator:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def _strata(self, gold, pred, report_ids, label_of: dict, labels, key) -> pd.DataFrame:
        rows = []
        for label in labels:
            ids = [rid for rid in report_ids if label_of.get(rid) == label]
            if not ids:
                continue
            sub = multiset_counts(gold[gold.report_id.isin(ids)], pred[pred.report_id.isin(ids)], ids, key)
            s = summarise(sub, self.config)
            rows.append({"group": label, "reports": len(ids), "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
                         "precision": s["precision"], "recall": s["recall"], "f1": s["f1"],
                         "precision_ci": s["ci95"]["precision"], "recall_ci": s["ci95"]["recall"],
                         "f1_ci": s["ci95"]["f1"]})
        return pd.DataFrame(rows)

    def evaluate(self, run: PipelineRun) -> EvaluationResult:
        cfg = self.config
        report_ids = sorted(r.report_id for r in run.reports)
        gold = prepare_gold(cfg.gold_csv)
        gold = gold[gold.report_id.isin(report_ids)].copy()
        pred = prepare_predictions(run)
        style_of = {r.report_id: r.demographics.style.value for r in run.reports if r.demographics}
        split_rows = _read_csv_dicts(cfg.splits_csv) if cfg.splits_csv.exists() else []
        source_of = {r["report_id"]: r["source"] for r in split_rows}
        split_of = {r["report_id"]: r["split"] for r in split_rows}

        concept_key = lambda row: (row["organ_norm"], row["concept"], row["status"])  # noqa: E731
        strict_key = lambda row: (row["organ_norm"], row["finding_strict"], row["status"])  # noqa: E731
        concept_tab = multiset_counts(gold, pred, report_ids, concept_key)
        strict_tab = multiset_counts(gold, pred, report_ids, strict_key)

        by_style = self._strata(gold, pred, report_ids, style_of, STYLES, concept_key)
        by_style = by_style.rename(columns={"group": "style"})
        by_source = self._strata(gold, pred, report_ids, source_of, sorted(set(source_of.values())), concept_key)
        by_source = by_source.rename(columns={"group": "source"})

        concept_rows = []
        pairs = pd.concat([gold[["organ_norm", "concept"]], pred[["organ_norm", "concept"]]]).drop_duplicates()
        for organ, concept in pairs.itertuples(index=False):
            g = gold[(gold.organ_norm == organ) & (gold.concept == concept)]
            p = pred[(pred.organ_norm == organ) & (pred.concept == concept)]
            t = multiset_counts(g, p, report_ids, concept_key)
            pr, rc, f1 = prf(t.tp.sum(), t.fp.sum(), t.fn.sum())
            concept_rows.append({"organ": organ, "concept": concept, "gold": len(g), "pred": len(p),
                                 "tp": int(t.tp.sum()), "fp": int(t.fp.sum()), "fn": int(t.fn.sum()),
                                 "precision": pr, "recall": rc, "f1": f1})
        by_concept = pd.DataFrame(concept_rows).sort_values(["organ", "gold"], ascending=[True, False])

        aligned = align_by_concept(gold, pred)
        has_pairs = len(aligned) > 0
        assertion = {
            "pairs": int(len(aligned)),
            "accuracy": float((aligned.gold_status == aligned.pred_status).mean()) if has_pairs else float("nan"),
            "macro_f1": macro_f1(aligned.gold_status.tolist(), aligned.pred_status.tolist()) if has_pairs else float("nan"),
            "cohen_kappa": cohen_kappa(aligned.gold_status.tolist(), aligned.pred_status.tolist()) if has_pairs else float("nan"),
            "confusion": pd.crosstab(aligned.gold_status, aligned.pred_status).to_dict() if has_pairs else {},
        }
        sized = aligned.dropna(subset=["gold_size", "pred_size"]) if has_pairs else aligned
        gold_sized = int(gold["size_mm"].notna().sum())
        size = {
            "gold_entities_with_size": gold_sized,
            "pairs_with_size": int(len(sized)),
            "coverage": len(sized) / gold_sized if gold_sized else float("nan"),
            "mae_mm": float((sized.gold_size - sized.pred_size).abs().mean()) if len(sized) else float("nan"),
            "exact_match_rate": float(((sized.gold_size - sized.pred_size).abs() < 1e-6).mean()) if len(sized) else float("nan"),
        }
        steat = aligned[(aligned.concept == "Steatosis") & (aligned.gold_status == "Affirmed")] if has_pairs else aligned
        grade = {
            "pairs": int(len(steat)),
            "gold_steatosis": int(((gold.concept == "Steatosis") & (gold.status == "Affirmed")).sum()),
            "accuracy": float((steat.gold_grade == steat.pred_grade).mean()) if len(steat) else float("nan"),
            "confusion": pd.crosstab(steat.gold_grade, steat.pred_grade).to_dict() if len(steat) else {},
        }
        exact_status = aligned[aligned.gold_status == aligned.pred_status] if has_pairs else aligned
        ontology = {
            "radlex_agreement": float((exact_status.gold_radlex.fillna("-") == exact_status.pred_radlex.fillna("-")).mean())
            if len(exact_status) else float("nan"),
            "icd10_agreement": float((exact_status.gold_icd.fillna("-") == exact_status.pred_icd.fillna("-")).mean())
            if len(exact_status) else float("nan"),
        }

        def token_recall(g: str, p: str) -> float:
            gt = set(re.findall(r"\w+", str(g).lower()))
            pt = set(re.findall(r"\w+", str(p).lower()))
            return len(gt & pt) / len(gt) if gt else 0.0

        span_recall = [token_recall(g, p) for g, p in zip(exact_status.get("gold_evidence", []),
                                                           exact_status.get("pred_evidence", []))]
        source = {r.report_id: r.deidentification.redacted_text for r in run.reports}
        positive = pred[pred.status != "Negated"]
        ungrounded = sum(ev not in source.get(rid, "") for rid, ev in zip(positive.report_id, positive.evidence_text))

        g_flags, p_flags = _report_flags(gold, report_ids), _report_flags(pred, report_ids)
        rng = np.random.default_rng(cfg.bootstrap_seed)
        boot = rng.integers(0, len(report_ids), size=(cfg.bootstrap_resamples, len(report_ids)))
        prev_rows = []
        for cond in g_flags.columns:
            gv, pv = g_flags[cond].to_numpy(float), p_flags[cond].to_numpy(float)
            if gv.sum() == 0 and pv.sum() == 0:
                continue
            err = np.abs(pv[boot].mean(1) - gv[boot].mean(1)) * 100
            prev_rows.append({"condition": cond, "gold_prevalence": gv.mean(), "pred_prevalence": pv.mean(),
                              "abs_error_pp": abs(pv.mean() - gv.mean()) * 100,
                              "abs_error_ci95_pp": [float(np.percentile(err, 2.5)), float(np.percentile(err, 97.5))]})
        prevalence = pd.DataFrame(prev_rows)
        g_tier, p_tier = _tier(gold, report_ids), _tier(pred, report_ids)
        tiers_g = [str(g_tier[r]) for r in report_ids]
        tiers_p = [str(p_tier[r]) for r in report_ids]

        splits_seen = sorted({split_of.get(r, "unknown") for r in report_ids})
        summary = {
            "run_name": cfg.run_name,
            "extractor": run.extractor_info,
            "split": cfg.split,
            "splits_evaluated": splits_seen,
            "n_reports": len(report_ids),
            "n_gold_entities": int(len(gold)),
            "n_pred_entities": int(len(pred)),
            "entity_concept": summarise(concept_tab, cfg),
            "entity_strict": summarise(strict_tab, cfg),
            "assertion": assertion,
            "size_mm": size,
            "steatosis_grade": grade,
            "ontology_grounding": ontology,
            "evidence_token_recall_mean": float(np.mean(span_recall)) if span_recall else float("nan"),
            "hallucination_rate": ungrounded / len(positive) if len(positive) else 0.0,
            "model_reliability": run.reliability(),
            "schema_validity_pass_rate": run.schema_validity_pass_rate,
            "schema_errors": run.schema_errors[:50],
            "triage_agreement": {
                "accuracy": float(np.mean([a == b for a, b in zip(tiers_g, tiers_p)])) if report_ids else float("nan"),
                "cohen_kappa": cohen_kappa(tiers_g, tiers_p),
            },
            "phi": evaluate_phi(run, cfg.phi_gold_csv),
            "prevalence": prevalence.to_dict(orient="records"),
            "by_style": by_style.to_dict(orient="records"),
            "by_source": by_source.to_dict(orient="records"),
            "throughput_reports_per_s": run.throughput,
            "nlp_seconds": run.nlp_seconds,
            "bootstrap": {"resamples": cfg.bootstrap_resamples, "seed": cfg.bootstrap_seed, "unit": "report"},
        }
        if "test" in splits_seen:
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            with cfg.test_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     "run_name": cfg.run_name, "extractor": run.extractor_info,
                                     "n_reports": len(report_ids),
                                     "f1_concept": summary["entity_concept"]["f1"]}) + "\n")
        errors = pd.DataFrame(concept_tab.errors, columns=["report_id", "error", "key"])
        if not errors.empty:
            errors["style"] = errors["report_id"].map(style_of)
            errors["source"] = errors["report_id"].map(source_of)
        else:
            errors = errors.assign(style=pd.Series(dtype=str), source=pd.Series(dtype=str))
        return EvaluationResult(summary, by_style, by_concept, aligned, errors, prevalence, by_source)

    def save(self, result: EvaluationResult) -> None:
        cfg = self.config
        cfg.ensure_output_dir()
        cfg.evaluation_json.write_text(result.to_json(), encoding="utf-8")
        result.errors.to_csv(cfg.error_analysis_csv, index=False)
        result.by_concept.to_csv(cfg.run_dir / "metrics_by_concept.csv", index=False)
        result.by_style.to_csv(cfg.run_dir / "metrics_by_style.csv", index=False)
        result.by_source.to_csv(cfg.run_dir / "metrics_by_source.csv", index=False)
        result.aligned.to_csv(cfg.run_dir / "aligned_entities.csv", index=False)
        result.prevalence.to_csv(cfg.run_dir / "prevalence_concordance.csv", index=False)
