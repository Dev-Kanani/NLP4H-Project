"""Vitals & Vectors - interactive clinical NLP dashboard.

Run from the project root::

    streamlit run dashboard/app.py

Pages
-----
1. Report Structurer      - live sectionization, colour-coded findings, assertion badges, CSV/JSON export
2. Population Analytics   - demographics, prevalence with Wilson CIs, cross-tabs, filterable cohort
3. Evaluation Suite       - benchmark tables, bootstrap CIs, error analysis
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import replace  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from usg_nlp.analytics import AGE_BAND_ORDER, CONDITIONS, PopulationAnalytics  # noqa: E402
from usg_nlp.config import PipelineConfig  # noqa: E402
from usg_nlp.database import ClinicalDatabase  # noqa: E402
from usg_nlp.evaluation import Evaluator  # noqa: E402
from usg_nlp.extractors import LLMExtractor  # noqa: E402
from usg_nlp.llm import PromptBuilder, make_backend, select_examples  # noqa: E402
from usg_nlp.pipeline import UltrasoundPipeline  # noqa: E402
from usg_nlp.schemas import StructuredReport  # noqa: E402
from usg_nlp.sectionizer import AnatomicalSectionizer  # noqa: E402

# --------------------------------------------------------------------------- design tokens
# Validated reference palette (dataviz skill): categorical slots 1-2 for sex,
# reserved status colours for assertion/triage, single-hue blue ramp for grade.
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
SEX_COLORS = {"Female": "#2a78d6", "Male": "#eb6834"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
TIER_COLORS = {1: STATUS["good"], 2: STATUS["warning"], 3: STATUS["critical"]}
TIER_ICONS = {1: "✓", 2: "●", 3: "▲"}
GRADE_COLORS = {"Grade 1": "#86b6ef", "Grade 2": "#3987e5", "Grade 3": "#184f95", "N/A": "#c3c2b7"}
NON_PATHOLOGICAL = {"Unremarkable", "Unremarkable Study", "Bowel Gas Obscuration"}

st.set_page_config(page_title="Vitals & Vectors · USG NLP", page_icon="🩺", layout="wide")
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.6rem; }}
      .badge {{ display:inline-block; padding:1px 8px; border-radius:10px; font-size:0.78rem;
               font-weight:600; border:1px solid rgba(11,11,11,.12); }}
      .report-box {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size:0.86rem;
                     line-height:1.6; white-space:pre-wrap; padding:14px 16px; border-radius:8px;
                     border:1px solid rgba(128,128,128,.25); }}
      .report-box mark {{ color: inherit; padding:1px 0; border-radius:3px; }}
      .tier-banner {{ padding:10px 14px; border-radius:8px; border-left:6px solid; margin-bottom:8px;
                      background: rgba(128,128,128,.06); }}
      .legend span {{ margin-right:14px; font-size:0.8rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- data access
CFG = PipelineConfig()


@st.cache_resource(show_spinner=False)
def get_pipeline() -> UltrasoundPipeline:
    """Rule-baseline pipeline used for live processing of pasted text."""
    return UltrasoundPipeline(CFG)


@st.cache_resource(show_spinner="Building the rule-baseline run on the dev split ...")
def ensure_baseline(version: int) -> None:
    """Create outputs/rules/ on first launch (or on 'Re-run baseline'); LLM runs come from run_pipeline.py."""
    cfg = replace(CFG, run_name="rules", split="dev")
    if version == 0 and cfg.predictions_json.exists():
        return
    run = UltrasoundPipeline(cfg).run(persist=True)
    evaluator = Evaluator(cfg)
    evaluator.save(evaluator.evaluate(run))


def list_runs() -> list[str]:
    runs = sorted(p.parent.name for p in CFG.output_dir.glob("*/structured_reports.json"))
    return sorted(runs, key=lambda r: (r != "rules", r))


@st.cache_data(show_spinner=False)
def load_run(run_name: str, stamp: float):
    run_dir = CFG.output_dir / run_name
    reports = [StructuredReport.model_validate(d)
               for d in json.loads((run_dir / "structured_reports.json").read_text(encoding="utf-8"))]
    evaluation = None
    if (run_dir / "evaluation_summary.json").exists():
        summary = json.loads((run_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
        def read(name: str) -> pd.DataFrame:
            path = run_dir / name
            if not path.exists() or path.stat().st_size == 0:
                return pd.DataFrame()
            try:
                return pd.read_csv(path)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()

        evaluation = SimpleNamespace(
            summary=summary, by_style=pd.DataFrame(summary.get("by_style", [])),
            by_source=pd.DataFrame(summary.get("by_source", [])), prevalence=pd.DataFrame(summary.get("prevalence", [])),
            by_concept=read("metrics_by_concept.csv"), errors=read("error_analysis.csv"))
    return reports, evaluation


@st.cache_resource(show_spinner="Connecting to the model ...")
def get_llm_pipeline(backend: str, model: str, base_url: str, few_shot: int):
    be = make_backend(backend, model, base_url=base_url or None)
    ok, msg = be.available()
    if not ok:
        return None, msg
    ext = LLMExtractor(be, PromptBuilder(examples=select_examples(CFG.data_dir, n=few_shot)))
    return UltrasoundPipeline(CFG, extractor=ext), "ok"


def style_of(report: StructuredReport) -> str:
    return AnatomicalSectionizer().sectionize(report.deidentification.redacted_text).style_detected


def gold_frame() -> pd.DataFrame | None:
    path = CFG.gold_csv
    return pd.read_csv(path) if path.exists() else None


def plot_layout(fig: go.Figure, height: int = 320, **kwargs) -> go.Figure:
    title = fig.layout.title.text
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=72, b=10), plot_bgcolor=SURFACE, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK_2, size=12),
        title=dict(text=title, x=0, xref="paper", y=0.985, yref="container", yanchor="top",
                   font=dict(color=INK, size=15)),
        hoverlabel=dict(bgcolor="white", font_color=INK), bargap=0.25, bargroupgap=0.08,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, traceorder="normal"), **kwargs,
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont_color=MUTED)
    return fig


def findings_df(report: StructuredReport) -> pd.DataFrame:
    rows = [f.model_dump(mode="json") for f in report.findings]
    cols = ["organ", "finding", "status", "grade", "size_mm", "laterality", "location",
            "radlex_id", "icd10_code", "triage_tier", "evidence_text", "flags"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def status_style(row: pd.Series) -> list[str]:
    pathological = row["finding"] not in NON_PATHOLOGICAL and not row["finding"].startswith("Unremarkable")
    if row["status"] == "Negated":
        color = "rgba(137,135,129,.12)"
    elif row["status"] == "Possible":
        color = "rgba(250,178,25,.28)"
    elif pathological:
        color = "rgba(208,59,59,.20)"
    else:
        color = "rgba(12,163,12,.14)"
    return [f"background-color: {color}"] * len(row)


def highlighted_report(text: str, report: StructuredReport) -> str:
    """Overlay evidence spans on the de-identified text, coloured by the most salient assertion."""
    rank = [0] * len(text)
    for f in report.findings:
        pathological = f.concept not in NON_PATHOLOGICAL
        level = {"Negated": 1, "Possible": 3}.get(f.status.value, 4 if pathological else 2)
        for i in range(max(0, f.span_start), min(len(text), f.span_end)):
            rank[i] = max(rank[i], level)
    colours = {1: "rgba(137,135,129,.22)", 2: "rgba(12,163,12,.20)", 3: "rgba(250,178,25,.40)",
               4: "rgba(208,59,59,.26)"}
    out, i = [], 0
    while i < len(text):
        j = i
        while j < len(text) and rank[j] == rank[i]:
            j += 1
        chunk = html.escape(text[i:j])
        out.append(f'<mark style="background:{colours[rank[i]]}">{chunk}</mark>' if rank[i] else chunk)
        i = j
    return "".join(out)


LEGEND_HTML = (
    '<div class="legend">'
    '<span><mark style="background:rgba(208,59,59,.26)">&nbsp;▲ Affirmed abnormality&nbsp;</mark></span>'
    '<span><mark style="background:rgba(250,178,25,.40)">&nbsp;? Possible&nbsp;</mark></span>'
    '<span><mark style="background:rgba(12,163,12,.20)">&nbsp;✓ Normal / unremarkable&nbsp;</mark></span>'
    '<span><mark style="background:rgba(137,135,129,.22)">&nbsp;– Negated&nbsp;</mark></span></div>'
)

# --------------------------------------------------------------------------- sidebar
if "run_version" not in st.session_state:
    st.session_state.run_version = 0
with st.sidebar:
    st.markdown("### 🩺 Vitals & Vectors")
    st.caption("Abdominal USG information extraction & population health analytics · NLP for Healthcare, IIIT Hyderabad")
    page = st.radio("Page", ["Report Structurer", "Population Analytics", "Evaluation Suite"], label_visibility="collapsed")
    st.divider()
    ensure_baseline(st.session_state.run_version)
    runs = list_runs()
    run_name = st.selectbox("Run (extractor / model)", runs,
                            help="Runs are created with run_pipeline.py; 'rules' is the frozen baseline.")
    with st.expander("Live LLM for pasted text"):
        llm_backend = st.selectbox("Backend", ["ollama", "openai"], help="openai = any OpenAI-compatible server, e.g. vLLM")
        llm_model = st.text_input("Model", "qwen2.5:7b-instruct")
        llm_url = st.text_input("Server URL (blank = default)", "")
        llm_shots = st.slider("Few-shot examples", 0, 5, 3)
    if st.button("Re-run rule baseline", width="stretch"):
        st.session_state.run_version += 1
        st.cache_data.clear()
    st.caption("Follow-up tiers are assistive administrative groupings, not clinical triage or diagnosis.")

run_dir = CFG.output_dir / run_name
loaded_reports, evaluation = load_run(run_name, (run_dir / "structured_reports.json").stat().st_mtime)
reports = {r.report_id: r for r in loaded_reports}

# =========================================================================== page 1
if page == "Report Structurer":
    st.title("Interactive Report Structurer")
    source = st.radio("Input", ["Cohort report", "Paste custom text"], horizontal=True)
    if source == "Cohort report":
        rid = st.selectbox("Report", sorted(reports), format_func=lambda r: (
            f"{r} · {reports[r].demographics.age}y {reports[r].demographics.sex} · {reports[r].demographics.style.value}"
            if reports[r].demographics else r))
        report = reports[rid]
        style = style_of(report)
        st.caption(f"Showing recorded output of run **{run_name}**.")
    else:
        sample = ("Liver: Grade-II fatty liver. Mildly enlarged liver.\nGB: Calculus cannot be ruled out due to "
                  "obscuring bowel gas. No pericholecystic fluid.\nPAN/SPL: WNL; no focal abnormality.\n"
                  "RK: Simple cortical cyst, upper pole, 14 mm. No hydro.\nLK: No calculus.\nPelvis: No free fluid.\n"
                  "Imp: Grade II fatty liver. Right renal cortical cyst.")
        text = st.text_area("Report text (identifiers are redacted before processing)", sample, height=210)
        engine = st.radio("Extractor", ["Rule baseline", "LLM (live)"], horizontal=True)
        pipe = get_pipeline()
        if engine == "LLM (live)":
            llm_pipe, msg = get_llm_pipeline(llm_backend, llm_model, llm_url, llm_shots)
            if llm_pipe is None:
                st.warning(f"{msg}. Falling back to the rule baseline.")
            else:
                pipe = llm_pipe
        with st.spinner("Extracting ..."):
            report, style, errs = pipe.process_text("CUSTOM_001", text)
        if errs:
            st.warning("Schema validation rejected: " + "; ".join(errs))

    tier = int(report.triage_tier)
    st.markdown(
        f'<div class="tier-banner" style="border-color:{TIER_COLORS[tier]}"><b>{TIER_ICONS[tier]} Tier {tier} · '
        f'{html.escape(report.triage_label)}</b><br><span style="font-size:.85rem">'
        f'{html.escape(" · ".join(report.triage_reasons))}</span></div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Detected style", style)
    m2.metric("Sections", len(report.sections))
    m3.metric("Findings", len(report.findings))
    m4.metric("Affirmed / Negated / Possible",
              " / ".join(str(sum(f.status.value == s for f in report.findings)) for s in ("Affirmed", "Negated", "Possible")))
    m5.metric("PHI redacted", report.deidentification.phi_count)

    left, right = st.columns([5, 7])
    with left:
        st.subheader("Source (de-identified) with evidence spans")
        st.markdown(f'<div class="report-box">{highlighted_report(report.deidentification.redacted_text, report)}</div>',
                    unsafe_allow_html=True)
        st.markdown(LEGEND_HTML, unsafe_allow_html=True)
        if report.deidentification.phi_entities:
            st.caption("Redacted identifiers: " + ", ".join(
                f"{e.phi_type}" for e in report.deidentification.phi_entities))
    with right:
        st.subheader("Sectionization")
        sec_df = pd.DataFrame([{"section": s.label, "method": s.method, "header": s.header or "—",
                                "text": s.text} for s in report.sections])
        st.dataframe(sec_df, hide_index=True, width="stretch", height=min(320, 38 + 35 * len(sec_df)))

    st.subheader("Structured findings")
    fdf = findings_df(report)
    if fdf.empty:
        st.info("No findings extracted.")
    else:
        st.dataframe(fdf.style.apply(status_style, axis=1), hide_index=True, width="stretch",
                     column_config={"size_mm": st.column_config.NumberColumn("size (mm)", format="%.1f"),
                                    "evidence_text": st.column_config.TextColumn("evidence", width="large")})
    d1, d2, _ = st.columns([1, 1, 4])
    d1.download_button("⬇ CSV", fdf.to_csv(index=False), f"{report.report_id}_findings.csv", "text/csv")
    d2.download_button("⬇ JSON", json.dumps(report.model_dump(mode="json"), indent=2),
                       f"{report.report_id}_structured.json", "application/json")

    gold = gold_frame()
    if source == "Cohort report" and gold is not None:
        with st.expander("Compare with gold-standard annotation"):
            st.dataframe(gold[gold.report_id == report.report_id][
                ["organ", "finding", "grade", "size_mm", "status", "radlex_id", "icd10_code", "evidence_text"]],
                hide_index=True, width="stretch")

# =========================================================================== page 2
elif page == "Population Analytics":
    st.title("Workforce Population Health Analytics")
    analytics = PopulationAnalytics(ClinicalDatabase(run_dir / "clinical_ultrasound.db"))
    st.caption(f"Run: **{run_name}**")
    cohort = analytics.cohort()

    f1, f2, f3, f4 = st.columns(4)
    sexes = f1.multiselect("Biological sex", ["Female", "Male"], ["Female", "Male"])
    bands = f2.multiselect("Age band", AGE_BAND_ORDER, AGE_BAND_ORDER)
    styles = f3.multiselect("Reporting style", sorted(cohort["style"].unique()), sorted(cohort["style"].unique()))
    tiers = f4.multiselect("Follow-up tier", [1, 2, 3], [1, 2, 3])
    sel = cohort[cohort.sex.isin(sexes) & cohort.age_band.isin(bands) & cohort["style"].isin(styles)
                 & cohort.triage_tier.isin(tiers)]
    if sel.empty:
        st.warning("No patients match the current filters.")
        st.stop()

    n = len(sel)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Employees screened", n)
    for col, (key, label) in zip((k2, k3, k4), (("steatosis", "Hepatic steatosis"), ("cholelithiasis", "Cholelithiasis"),
                                                ("renal_cyst", "Renal cortical cyst"))):
        k = int(sel[key].sum())
        col.metric(label, f"{k / n:.0%}", f"{k} of {n}", delta_color="off", delta_arrow="off")
    review = int((sel.triage_tier == 3).sum())
    k5.metric("▲ Clinician review", f"{review / n:.0%}", f"{review} of {n}", delta_color="off", delta_arrow="off")
    st.caption(f"Cohort n = {n}. Subgroup cells are small - read prevalence with its interval; associations are "
               "descriptive, not causal.")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for sex in sexes:
            ages = sel[sel.sex == sex].age
            fig.add_histogram(x=ages, name=sex, marker_color=SEX_COLORS[sex], xbins=dict(start=20, end=65, size=5),
                              marker_line=dict(color=SURFACE, width=2),
                              hovertemplate="%{x} y: %{y} employees<extra>" + sex + "</extra>")
        fig.update_layout(barmode="stack", title="Age distribution by sex")
        fig.update_xaxes(title="Age (years)")
        fig.update_yaxes(title="Employees")
        st.plotly_chart(plot_layout(fig), width="stretch")
    with c2:
        prev = analytics.prevalence(sel)
        prev = prev[prev.condition != "Any abnormality"].sort_values("prevalence")
        fig = go.Figure(go.Bar(
            x=prev.prevalence, y=prev.condition, orientation="h", marker_color="#86b6ef", width=0.55,
            error_x=dict(type="data", symmetric=False, array=prev.ci_high - prev.prevalence,
                         arrayminus=prev.prevalence - prev.ci_low, color=INK_2, thickness=1.4, width=5),
            customdata=prev[["cases", "n", "ci_low", "ci_high"]],
            hovertemplate="%{y}<br>%{customdata[0]} of %{customdata[1]} · %{x:.0%}"
                          "<br>Wilson 95% CI %{customdata[2]:.0%}-%{customdata[3]:.0%}<extra></extra>"))
        for row in prev.itertuples():  # label sits past the CI whisker so the two never collide
            fig.add_annotation(x=row.ci_high + 0.02, y=row.condition, xanchor="left", showarrow=False,
                               text=f"{row.prevalence:.0%} · {row.cases}/{row.n}", font=dict(color=INK_2, size=12))
        fig.update_layout(title="Prevalence · whisker = Wilson 95% CI")
        fig.update_xaxes(tickformat=".0%", range=[0, 1.08], title=None, showgrid=True, gridcolor=GRID)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(plot_layout(fig), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        condition = st.selectbox("Condition for stratified view", [k for k in CONDITIONS if k != "any_abnormality"],
                                 format_func=CONDITIONS.get)
        by = analytics.prevalence(sel, by=["age_band", "sex"])
        by = by[by.condition == CONDITIONS[condition]]
        by["band"] = [g[0] for g in by.group]
        by["sex"] = [g[1] for g in by.group]
        fig = go.Figure()
        for sex in sexes:
            d = by[by.sex == sex]
            fig.add_bar(x=d.band.astype(str), y=d.prevalence, name=sex, marker_color=SEX_COLORS[sex],
                        customdata=d[["cases", "n"]],
                        hovertemplate="%{x} · " + sex + "<br>%{customdata[0]} of %{customdata[1]} (%{y:.0%})<extra></extra>")
        fig.update_layout(barmode="group", title=f"{CONDITIONS[condition]} by age band and sex")
        fig.update_xaxes(categoryorder="array", categoryarray=AGE_BAND_ORDER, title="Age band")
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        st.plotly_chart(plot_layout(fig), width="stretch")
        st.markdown("**Cross-tabulation (cases / employees)**")
        st.dataframe(analytics.crosstab(sel, condition), width="stretch")
    with c4:
        tier_counts = sel.triage_tier.value_counts().reindex([1, 2, 3]).fillna(0).astype(int)
        labels = {1: "Unremarkable", 2: "Routine preventive tracking", 3: "Clinician review"}
        fig = go.Figure(go.Bar(
            x=[f"{TIER_ICONS[t]} {labels[t]}" for t in tier_counts.index], y=tier_counts.values,
            marker_color=[TIER_COLORS[t] for t in tier_counts.index], text=tier_counts.values,
            textposition="outside", textfont_color=INK_2, cliponaxis=False,
            hovertemplate="%{x}: %{y} employees<extra></extra>"))
        fig.update_layout(title="Follow-up tier distribution", showlegend=False)
        st.plotly_chart(plot_layout(fig), width="stretch")
        grades = analytics.steatosis_grades(sel)
        grades = grades[grades.patients > 0]
        fig = go.Figure(go.Bar(
            x=grades.grade, y=grades.patients, marker_color=[GRADE_COLORS[g] for g in grades.grade],
            text=grades.patients, textposition="outside", textfont_color=INK_2, cliponaxis=False,
            hovertemplate="%{x}: %{y} employees<extra></extra>"))
        fig.update_layout(title="Steatosis severity (affected employees)", showlegend=False)
        st.plotly_chart(plot_layout(fig, height=260), width="stretch")

    with st.expander("Condition co-occurrence (employees with both)"):
        st.dataframe(analytics.co_occurrence(sel), width="stretch")

    st.subheader("Patient cohort")
    table = sel[["report_id", "age", "sex", "age_band", "style", "triage_tier", "triage_label", "steatosis",
                 "steatosis_grade", "hepatomegaly", "cholelithiasis", "renal_cyst", "possible_renal_calculus"]].copy()
    for col in ("steatosis", "hepatomegaly", "cholelithiasis", "renal_cyst", "possible_renal_calculus"):
        table[col] = table[col].astype(bool)
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("⬇ Cohort CSV", table.to_csv(index=False), "cohort_conditions.csv", "text/csv")

# =========================================================================== page 3
else:
    st.title("Evaluation Suite")
    if evaluation is None:
        st.error("Gold standard CSV not found - nothing to evaluate.")
        st.stop()
    s = evaluation.summary
    st.caption(f"{s['n_reports']} reports · {s['n_gold_entities']} gold entities · {s['n_pred_entities']} predicted · "
               f"{s['bootstrap']['resamples']} cluster-bootstrap resamples over reports (seed {s['bootstrap']['seed']})")
    st.markdown(f"Run **{run_name}** · extractor `{s['extractor'].get('model', s['extractor'].get('extractor'))}` · "
                f"split **{s['split']}**")
    if "test" in s.get("splits_evaluated", []):
        st.info("Includes the locked test split (076-100) - this is a final-comparison number.", icon="🔒")
    else:
        st.warning("Development split. The rule baseline was written with reports 001-050 in view, so its "
                   "'original' row is an upper bound; the 'v2_generated' row is the honest dev estimate.", icon="⚠️")

    comp = []
    for path in sorted(CFG.output_dir.glob("*/evaluation_summary.json")):
        x = json.loads(path.read_text())
        src = {r["source"]: r["f1"] for r in x.get("by_source", [])}
        comp.append({"run": x["run_name"], "split": x["split"], "reports": x["n_reports"],
                     "F1": round(x["entity_concept"]["f1"], 3),
                     "F1 95% CI": f"{x['entity_concept']['ci95']['f1'][0]:.3f}-{x['entity_concept']['ci95']['f1'][1]:.3f}",
                     "F1 original": round(src.get("original", float("nan")), 3),
                     "F1 v2": round(src.get("v2_generated", float("nan")), 3),
                     "hallucination": round(x["model_reliability"]["hallucination_rate"], 3),
                     "seconds": round(x["nlp_seconds"], 1)})
    if len(comp) > 1:
        st.subheader("Run comparison")
        st.dataframe(pd.DataFrame(comp), hide_index=True, width="stretch")

    ec = s["entity_concept"]
    m = st.columns(6)
    for col, key in zip(m[:3], ("precision", "recall", "f1")):
        lo, hi = ec["ci95"][key]
        col.metric(f"Micro {key.upper() if key == 'f1' else key.title()}", f"{ec[key]:.3f}",
                   f"95% CI {lo:.3f}-{hi:.3f}", delta_color="off", delta_arrow="off")
    m[3].metric("Assertion accuracy", f"{s['assertion']['accuracy']:.3f}", f"κ = {s['assertion']['cohen_kappa']:.3f}",
                delta_color="off", delta_arrow="off")
    m[4].metric("Size MAE", f"{s['size_mm']['mae_mm']:.2f} mm",
                f"{s['size_mm']['pairs_with_size']}/{s['size_mm']['gold_entities_with_size']} sizes aligned", delta_color="off", delta_arrow="off")
    m[5].metric("Steatosis grade acc.", f"{s['steatosis_grade']['accuracy']:.3f}",
                f"{s['steatosis_grade']['pairs']}/{s['steatosis_grade']['gold_steatosis']} aligned", delta_color="off", delta_arrow="off")

    rows = []
    for mode, label in (("entity_concept", "Concept-normalised (primary)"), ("entity_strict", "Strict surface string")):
        e = s[mode]
        rows.append({"matching": label, "TP": e["tp"], "FP": e["fp"], "FN": e["fn"],
                     **{k: f"{e[k]:.4f} [{e['ci95'][k][0]:.3f}, {e['ci95'][k][1]:.3f}]" for k in ("precision", "recall", "f1")}})
    st.subheader("Entity-level extraction")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Style-stratified performance")
    bs = evaluation.by_style
    fig = go.Figure()
    metric_colors = {"precision": "#2a78d6", "recall": "#eb6834", "f1": "#1baf7a"}
    for metric, color in metric_colors.items():
        ci = bs[f"{metric}_ci"].tolist()
        fig.add_scatter(
            x=bs["style"], y=bs[metric], name=metric.upper() if metric == "f1" else metric.title(), mode="markers",
            marker=dict(color=color, size=10, line=dict(color=SURFACE, width=2)),
            error_y=dict(type="data", symmetric=False, array=[c[1] - v for c, v in zip(ci, bs[metric])],
                         arrayminus=[v - c[0] for c, v in zip(ci, bs[metric])], color=color, thickness=1.5, width=5),
            hovertemplate="%{x} · " + metric + " %{y:.3f}<extra></extra>")
    fig.update_layout(scattermode="group", title="Point estimate with 95% bootstrap CI")
    lo = min(min(c[0] for c in bs[f"{m}_ci"]) for m in metric_colors)
    fig.update_yaxes(range=[max(0, lo - 0.02), 1.005], tickformat=".2f")
    c1, c2 = st.columns([1, 1])
    c1.plotly_chart(plot_layout(fig, height=330), width="stretch")
    show = bs.copy()
    for k in ("precision", "recall", "f1"):
        show[k] = [f"{v:.3f} [{c[0]:.3f}, {c[1]:.3f}]" for v, c in zip(show[k], show[f"{k}_ci"])]
    c2.dataframe(show[["style", "reports", "tp", "fp", "fn", "precision", "recall", "f1"]], hide_index=True, width="stretch")

    if not evaluation.by_source.empty:
        st.subheader("By corpus source")
        src = evaluation.by_source.copy()
        for k in ("precision", "recall", "f1"):
            src[k] = [f"{v:.3f} [{c[0]:.3f}, {c[1]:.3f}]" for v, c in zip(src[k], src[f"{k}_ci"])]
        st.dataframe(src[["source", "reports", "tp", "fp", "fn", "precision", "recall", "f1"]], hide_index=True,
                     width="stretch")
        st.caption("original = reports 001-050 (v1 labels); v2_generated = 051-100 (independently written, v2.1 "
                   "guideline). Labelling conventions differ slightly - see data/ANNOTATION_GUIDELINE.md §8.")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Assertion confusion (gold rows × predicted columns)")
        conf = pd.DataFrame(s["assertion"]["confusion"]).fillna(0).astype(int)
        st.dataframe(conf.T if not conf.empty else conf, width="stretch")
        st.caption(f"Macro-F1 {s['assertion']['macro_f1']:.3f} on {s['assertion']['pairs']} aligned entities.")
        st.subheader("Steatosis grade confusion")
        gconf = pd.DataFrame(s["steatosis_grade"]["confusion"]).fillna(0).astype(int)
        st.dataframe(gconf.T if not gconf.empty else gconf, width="stretch")
    with c4:
        st.subheader("Reliability & concordance")
        rel = pd.DataFrame([
            ("Schema validity pass rate (SVPR)", f"{s['schema_validity_pass_rate']:.4f}"),
            ("Hallucination rate (ungrounded positive findings)", f"{s['hallucination_rate']:.4f}"),
            ("Evidence token recall vs gold span", f"{s['evidence_token_recall_mean']:.4f}"),
            ("RadLex agreement (status-matched pairs)", f"{s['ontology_grounding']['radlex_agreement']:.4f}"),
            ("ICD-10-CM agreement (status-matched pairs)", f"{s['ontology_grounding']['icd10_agreement']:.4f}"),
            ("Report follow-up tier accuracy", f"{s['triage_agreement']['accuracy']:.4f}"),
            ("Throughput (reports / s)", f"{s['throughput_reports_per_s']:.1f}"),
        ], columns=["metric", "value"])
        rel_m = s["model_reliability"]
        rel = pd.concat([rel, pd.DataFrame([
            ("Model proposals / accepted", f"{rel_m['proposed']} / {rel_m['accepted']}"),
            ("Rejected: ungrounded evidence (hallucination)", f"{rel_m['hallucination_rate']:.4f}"),
            ("Rejected: out of vocabulary", f"{rel_m['out_of_vocabulary_rate']:.4f}"),
            ("Reports with no parsable output", str(rel_m["reports_without_parsable_output"])),
            ("Verifier flags", ", ".join(f"{k}: {v}" for k, v in rel_m["flags"].items()) or "none"),
        ], columns=["metric", "value"])], ignore_index=True)
        phi = s.get("phi", {})
        if phi.get("available") and phi.get("gold_identifiers"):
            rel = pd.concat([rel, pd.DataFrame([("De-identification: identifiers covered / false positives",
                                                 f"{phi['covered']}/{phi['gold_identifiers']} / {phi['false_positives']}")],
                                               columns=["metric", "value"])], ignore_index=True)
        st.dataframe(rel, hide_index=True, width="stretch")
        st.subheader("Prevalence estimation error")
        prev = evaluation.prevalence.copy()
        prev["gold"] = (prev.gold_prevalence * 100).map("{:.1f}%".format)
        prev["predicted"] = (prev.pred_prevalence * 100).map("{:.1f}%".format)
        prev["|p̂ − p| (pp)"] = prev.abs_error_pp.map("{:.1f}".format)
        st.dataframe(prev[["condition", "gold", "predicted", "|p̂ − p| (pp)"]], hide_index=True, width="stretch")

    st.subheader("Per-concept performance")
    bc = evaluation.by_concept.copy()
    for k in ("precision", "recall", "f1"):
        bc[k] = bc[k].round(3)
    st.dataframe(bc, hide_index=True, width="stretch")

    st.subheader("Error analysis")
    errs = evaluation.errors
    if errs.empty or "style" not in errs:
        st.success("No entity-level errors.")
    else:
        opts = sorted(errs["style"].dropna().unique())
        style_filter = st.multiselect("Style", opts, opts)
        st.dataframe(errs[errs["style"].isin(style_filter)], hide_index=True, width="stretch")
    with st.expander("Matching protocol"):
        st.markdown(
            "- **Entity** = (report, organ, finding, status); multiset matching per report, so a telegraphic report "
            "that negates hydronephrosis under both `RK:` and `LK:` needs two predictions.\n"
            "- **Concept mode** normalises annotation-surface variants (laterality suffixes, *Intrahepatic Biliary "
            "Dilatation* ≡ *Biliary Dilatation*, pancreatic *Focal Abnormality* ≡ *Focal Lesion*, *Unremarkable "
            "Visualized Portion* ≡ *Unremarkable*; whole-study normality aligned to organ `Global`).\n"
            "- **Strict mode** requires the exact annotated finding string.\n"
            "- CIs: percentile cluster bootstrap over reports (findings within a report are correlated).")
