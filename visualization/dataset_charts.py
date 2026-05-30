"""Dataset-level health charts for PrimitiveQA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from core.types import AnalysisResult, PrimitiveType

ALL_PRIMITIVES = [
    PrimitiveType.REACH,
    PrimitiveType.GRASP,
    PrimitiveType.LIFT,
    PrimitiveType.TRANSPORT,
    PrimitiveType.PLACE,
    PrimitiveType.RETRACT,
]

PRIMITIVE_COLORS = {
    PrimitiveType.REACH:     "#4C9BE8",
    PrimitiveType.GRASP:     "#F4A261",
    PrimitiveType.LIFT:      "#2A9D8F",
    PrimitiveType.TRANSPORT: "#E9C46A",
    PrimitiveType.PLACE:     "#E76F51",
    PrimitiveType.RETRACT:   "#8ECAE6",
    PrimitiveType.UNKNOWN:   "#CCCCCC",
}

SCORE_COLOR = "#4C9BE8"
WARN_COLOR  = "#E76F51"
OK_COLOR    = "#2A9D8F"


def dataset_health_score(results: list[AnalysisResult]) -> float:
    """Single 0-1 score for the dataset."""
    if not results:
        return 0.0
    composites = [r.overall_quality.composite for r in results]
    coverage = primitive_coverage_scores(results)
    named = [p for p in ALL_PRIMITIVES if p != PrimitiveType.UNKNOWN]
    cov_score = sum(1 for p in named if coverage.get(p, 0) > 0) / len(named)
    return float(np.clip(0.6 * np.mean(composites) + 0.4 * cov_score, 0, 1))


def primitive_coverage_scores(results: list[AnalysisResult]) -> dict[PrimitiveType, float]:
    """Fraction of clips that contain each primitive (at least once)."""
    n = max(len(results), 1)
    counts: dict[PrimitiveType, int] = {p: 0 for p in ALL_PRIMITIVES}
    for r in results:
        seen = {ss.segment.primitive for ss in r.segments}
        for p in ALL_PRIMITIVES:
            if p in seen:
                counts[p] += 1
    return {p: counts[p] / n for p in ALL_PRIMITIVES}


def summary_table(results: list[AnalysisResult], names: list[str]) -> pd.DataFrame:
    rows = []
    for name, r in zip(names, results):
        pts = r.trajectory.points
        det = sum(1 for p in pts if p.confidence > 0)
        det_pct = f"{100 * det // max(len(pts), 1)}%"
        primitives_found = sorted({
            ss.segment.primitive.value
            for ss in r.segments
            if ss.segment.primitive != PrimitiveType.UNKNOWN
        })
        q = r.overall_quality
        rows.append({
            "Clip": name,
            "Detection": det_pct,
            "Composite": round(q.composite, 2),
            "Smoothness": round(q.smoothness, 2),
            "Path Eff.": round(q.path_efficiency, 2),
            "Decisive": round(q.decisiveness, 2),
            "Primitives Found": ", ".join(primitives_found) if primitives_found else "none",
        })
    return pd.DataFrame(rows)


def dataset_radar(results: list[AnalysisResult]) -> go.Figure:
    """Average radar across all clips."""
    if not results:
        return go.Figure()

    def avg(attr: str) -> float:
        return float(np.mean([getattr(r.overall_quality, attr) for r in results]))

    categories = ["Smoothness", "Path Efficiency", "Decisiveness", "Detection Conf"]
    values = [avg("smoothness"), avg("path_efficiency"), avg("decisiveness"), avg("confidence_mean")]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(76,155,232,0.25)",
        line=dict(color=SCORE_COLOR, width=2),
        name="Dataset avg",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=320,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def primitive_coverage_chart(results: list[AnalysisResult]) -> go.Figure:
    """Bar chart: % of clips containing each primitive."""
    coverage = primitive_coverage_scores(results)
    labels = [p.value.capitalize() for p in ALL_PRIMITIVES]
    values = [coverage[p] * 100 for p in ALL_PRIMITIVES]
    colors = [
        OK_COLOR if coverage[p] > 0.3 else (
            "#E9C46A" if coverage[p] > 0 else WARN_COLOR
        )
        for p in ALL_PRIMITIVES
    ]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.0f}%" for v in values],
        textposition="outside",
    ))
    fig.add_hline(y=30, line_dash="dot", line_color="#AAAAAA",
                  annotation_text="30% threshold", annotation_position="right")
    fig.update_layout(
        yaxis=dict(range=[0, 110], title="% of clips"),
        xaxis_title="Primitive",
        height=300,
        margin=dict(l=10, r=10, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Primitive coverage across dataset",
    )
    return fig


def score_distribution(results: list[AnalysisResult], names: list[str]) -> go.Figure:
    """Horizontal bar chart of composite scores per clip, sorted."""
    paired = sorted(zip(names, results), key=lambda x: x[1].overall_quality.composite)
    sorted_names = [p[0] for p in paired]
    scores = [p[1].overall_quality.composite for p in paired]
    colors = [OK_COLOR if s >= 0.65 else ("#E9C46A" if s >= 0.4 else WARN_COLOR) for s in scores]

    fig = go.Figure(go.Bar(
        x=scores,
        y=sorted_names,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.2f}" for s in scores],
        textposition="outside",
    ))
    fig.add_vline(x=0.65, line_dash="dot", line_color=OK_COLOR,
                  annotation_text="Good (0.65)", annotation_position="top right")
    fig.add_vline(x=0.4, line_dash="dot", line_color=WARN_COLOR,
                  annotation_text="Low (0.4)", annotation_position="bottom right")
    fig.update_layout(
        xaxis=dict(range=[0, 1.1], title="Composite score"),
        height=max(200, 60 + 40 * len(results)),
        margin=dict(l=10, r=80, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Per-clip composite scores",
    )
    return fig


def missing_primitives_warning(results: list[AnalysisResult]) -> str:
    """HTML string flagging primitives with zero coverage."""
    coverage = primitive_coverage_scores(results)
    missing = [p.value for p in ALL_PRIMITIVES if coverage[p] == 0]
    low = [p.value for p in ALL_PRIMITIVES if 0 < coverage[p] < 0.3]

    if not missing and not low:
        return f'<span style="color:{OK_COLOR}">✅ All 6 primitives represented in the dataset.</span>'

    parts = []
    if missing:
        m = ", ".join(missing)
        parts.append(f'<span style="color:{WARN_COLOR}">❌ Missing primitives: <b>{m}</b> — no examples in dataset</span>')
    if low:
        l = ", ".join(low)
        parts.append(f'<span style="color:#E9C46A">⚠️ Under-represented: <b>{l}</b> — fewer than 30% of clips</span>')
    return "<br>".join(parts)
