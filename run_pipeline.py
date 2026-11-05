#!/usr/bin/env python3
"""Command-line runner: ingest -> de-identify -> extract -> verify -> store -> evaluate.

Rule baseline (CPU, ~0.1 s)::

    python run_pipeline.py --split dev

Open-weight LLM extractor (choose one way to serve the model)::

    # laptop / workstation with Ollama
    ollama pull qwen2.5:7b-instruct
    python run_pipeline.py --extractor llm --backend ollama --model qwen2.5:7b-instruct --split dev

    # Ada cluster with vLLM (OpenAI-compatible server)
    vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 16384 &
    python run_pipeline.py --extractor llm --backend openai --base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen2.5-7B-Instruct --split dev --workers 8

    # reproduce a recorded run from the response cache, no model needed
    python run_pipeline.py --extractor llm --backend replay --model Qwen/Qwen2.5-7B-Instruct --split dev

Compare every evaluated run::

    python run_pipeline.py --compare

The test split (076-100) is for the final comparison only; every evaluation
that touches it is logged to outputs/test_set_evaluations.log.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

from usg_nlp.config import PipelineConfig
from usg_nlp.evaluation import Evaluator
from usg_nlp.extractors import LLMExtractor, RuleBasedExtractor
from usg_nlp.llm import PromptBuilder, ResponseCache, make_backend, select_examples
from usg_nlp.pipeline import UltrasoundPipeline


def _fmt_ci(ci: list[float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def print_report(summary: dict) -> None:
    line = "=" * 78
    print(line)
    title = f"{summary['run_name']}  |  split={summary['split']}  |  {summary['extractor'].get('model', 'rules')}"
    print(title.center(78))
    print(line)
    print(f"Reports: {summary['n_reports']}   gold entities: {summary['n_gold_entities']}   "
          f"predicted: {summary['n_pred_entities']}")
    for mode in ("entity_concept", "entity_strict"):
        m = summary[mode]
        print(f"\n[{mode}]  TP={m['tp']} FP={m['fp']} FN={m['fn']}")
        for k in ("precision", "recall", "f1"):
            print(f"   {k:<10} {m[k]:.4f}  95% CI {_fmt_ci(m['ci95'][k])}")
    for title, rows, key in (("style", summary["by_style"], "style"), ("corpus source", summary["by_source"], "source")):
        if not rows:
            continue
        print(f"\n[by {title}, concept mode]")
        print(f"   {key:<14}{'n':>4}{'P':>9}{'R':>9}{'F1':>9}   F1 95% CI")
        for row in rows:
            print(f"   {row[key]:<14}{row['reports']:>4}{row['precision']:>9.4f}{row['recall']:>9.4f}"
                  f"{row['f1']:>9.4f}   {_fmt_ci(row['f1_ci'])}")
    a, s, g = summary["assertion"], summary["size_mm"], summary["steatosis_grade"]
    print(f"\n[assertion]   pairs={a['pairs']}  accuracy={a['accuracy']:.4f}  macro-F1={a['macro_f1']:.4f}  "
          f"kappa={a['cohen_kappa']:.4f}")
    print(f"[size_mm]     pairs={s['pairs_with_size']}/{s['gold_entities_with_size']}  MAE={s['mae_mm']:.3f} mm")
    print(f"[steatosis]   grade accuracy={g['accuracy']:.4f} on {g['pairs']}/{g['gold_steatosis']} aligned")
    t = summary["triage_agreement"]
    print(f"[triage]      report-tier accuracy={t['accuracy']:.4f}  kappa={t['cohen_kappa']:.4f}")
    phi = summary["phi"]
    if phi.get("available") and phi["gold_identifiers"]:
        print(f"[de-id]       identifiers covered {phi['covered']}/{phi['gold_identifiers']}  "
              f"false positives {phi['false_positives']}")
    rel = summary["model_reliability"]
    print(f"[reliability] SVPR={summary['schema_validity_pass_rate']:.4f}  proposed={rel['proposed']}  "
          f"accepted={rel['accepted']}  rejected={rel['rejected_by_reason']}")
    print(f"              hallucination={rel['hallucination_rate']:.4f}  OOV={rel['out_of_vocabulary_rate']:.4f}  "
          f"unparsable reports={rel['reports_without_parsable_output']}  retries={rel['retries']}  flags={rel['flags']}")
    if summary["prevalence"]:
        print("\n[prevalence concordance]")
        for row in summary["prevalence"]:
            print(f"   {row['condition']:<26} gold={row['gold_prevalence']*100:5.1f}%  "
                  f"pred={row['pred_prevalence']*100:5.1f}%  |err|={row['abs_error_pp']:.1f} pp")
    print(f"\n[throughput]  {summary['throughput_reports_per_s']:.1f} reports/s "
          f"({summary['nlp_seconds']:.2f} s extraction for {summary['n_reports']} reports)")
    if "test" in summary["splits_evaluated"]:
        print("NOTE: test-split reports were scored; this evaluation was logged to outputs/test_set_evaluations.log")
    print(line)


def compare_runs(output_dir: Path) -> int:
    rows = []
    for path in sorted(output_dir.glob("*/evaluation_summary.json")):
        s = json.loads(path.read_text())
        e = s["entity_concept"]
        rows.append((s["run_name"], s["split"], s["n_reports"], e["precision"], e["recall"], e["f1"], e["ci95"]["f1"],
                     s["model_reliability"]["hallucination_rate"], s["nlp_seconds"]))
    if not rows:
        print("No evaluated runs found.")
        return 1
    print(f"{'run':<34}{'split':<6}{'n':>4}{'P':>8}{'R':>8}{'F1':>8}  {'F1 95% CI':<16}{'halluc.':>8}{'sec':>9}")
    for r in rows:
        print(f"{r[0]:<34}{r[1]:<6}{r[2]:>4}{r[3]:>8.3f}{r[4]:>8.3f}{r[5]:>8.3f}  {_fmt_ci(r[6]):<16}{r[7]:>8.3f}{r[8]:>9.2f}")
    return 0


def build_extractor(args, config: PipelineConfig):
    if args.extractor == "rules":
        return RuleBasedExtractor(), "rules"
    examples = select_examples(config.data_dir, n=args.few_shot)
    prompts = PromptBuilder(examples=examples)
    identity_model = args.model
    cache_name = "".join(c if c.isalnum() or c in "._-" else "-" for c in identity_model)
    cache = None if args.no_cache else ResponseCache(config.llm_cache_dir / f"{cache_name}.jsonl")
    if args.backend == "replay":
        if cache is None or len(cache) == 0:
            raise SystemExit(f"No recorded responses for {identity_model} in {config.llm_cache_dir}")
        recorded = json.loads(next(l for l in (config.llm_cache_dir / f"{cache_name}.jsonl").read_text().splitlines()
                                   if l.strip()))["meta"]["model"]
        backend = make_backend("replay", identity_model, cache=cache, replay_identity=recorded)
    else:
        backend = make_backend(args.backend, args.model, base_url=args.base_url, api_key=args.api_key,
                               temperature=args.temperature, max_tokens=args.max_tokens)
        ok, msg = backend.available()
        if not ok:
            raise SystemExit(f"Model backend unavailable: {msg}")
    extractor = LLMExtractor(backend, prompts, cache=cache, max_retries=args.retries,
                             use_schema=not args.no_schema)
    print(f"LLM extractor: {backend.identity}  few-shot={[e.report_id for e in examples]}  cache="
          f"{'off' if cache is None else cache.path}")
    return extractor, args.run_name or extractor.run_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=["all", "dev", "test"], default="dev",
                        help="Reports to process (default dev; use test only for the final comparison)")
    parser.add_argument("--extractor", choices=["rules", "llm"], default="rules")
    parser.add_argument("--backend", choices=["ollama", "openai", "transformers", "replay"], default="ollama")
    parser.add_argument("--model", default="qwen2.5:7b-instruct",
                        help="e.g. qwen2.5:7b-instruct (Ollama), Qwen/Qwen2.5-7B-Instruct (vLLM/HF), "
                             "meta-llama/Llama-3.1-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--base-url", help="Server URL (Ollama default :11434, OpenAI-compatible default :8000/v1)")
    parser.add_argument("--api-key")
    parser.add_argument("--few-shot", type=int, default=3, help="Dev-split examples in the prompt (0-5)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1, help="Parallel requests (vLLM handles batching)")
    parser.add_argument("--no-schema", action="store_true", help="Disable grammar-constrained decoding")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--run-name", help="Output folder name under outputs/ (default derived from model)")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--compare", action="store_true", help="Print a comparison of all evaluated runs and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    config = PipelineConfig()
    if args.data_dir:
        config = replace(config, data_dir=args.data_dir.resolve())
    if args.output_dir:
        config = replace(config, output_dir=args.output_dir.resolve())
    if args.compare:
        return compare_runs(config.output_dir)
    config = replace(config, bootstrap_resamples=args.bootstrap, split=args.split)
    if args.split == "test":
        print("You are evaluating on the locked TEST split. Do this once per final system; the run is logged.")

    extractor, run_name = build_extractor(args, config)
    config = replace(config, run_name=run_name)

    t0 = time.perf_counter()
    run = UltrasoundPipeline(config, extractor=extractor, workers=args.workers).run(persist=True)
    print(f"Pipeline [{run_name}]: {len(run.reports)} reports, {run.n_findings} findings, "
          f"{run.nlp_seconds:.2f}s extraction, SVPR={run.schema_validity_pass_rate:.3f}")
    for w in run.ingestion_warnings:
        print(f"  ingestion warning: {w}")

    if not args.no_eval:
        if not config.gold_csv.exists():
            print(f"Gold standard not found at {config.gold_csv}; skipping evaluation.", file=sys.stderr)
        else:
            evaluator = Evaluator(config)
            result = evaluator.evaluate(run)
            evaluator.save(result)
            print_report(result.summary)

    print(f"Artefacts written to {config.run_dir}  (end-to-end {time.perf_counter() - t0:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
