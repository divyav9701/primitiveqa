"""Plotly charts for PrimitiveQA."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from core.types import AnalysisResult, PrimitiveType, ScoredSegment

PRIMITIVE_COLORS = {
    PrimitiveType.REACH:     "#4C9BE8",
    PrimitiveType.GRASP:     "#F4A261",
    PrimitiveType.LIFT:      "#2A9D8F",
    PrimitiveType.TRANSPORT: "#E9C46A",
    PrimitiveType.PLACE:     "#E76F51",
    PrimitiveType.RETRACT:   "#8ECAE6",
    PrimitiveType.UNKNOWN:   "#AAAAAA",
}


def quality_radar(result: AnalysisResult) -> go.Figure:
    """Radar chart of overall quality dimensions."""
    q = result.overall_quality
    categories = ["Smoothness", "Path Efficiency", "Decisiveness", "Detection Conf"]
    values = [q.smoothness, q.path_efficiency, q.decisiveness, q.confidence_mean]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(76, 155, 232, 0.3)",
        line=dict(color="#4C9BE8", width=2),
        name="Your clip",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def primitive_timeline(scored: list[ScoredSegment], total_frames: int) -> go.Figure:
    """Gantt-style horizontal bar chart of primitive segments."""
    fig = go.Figure()

    for ss in scored:
        seg = ss.segment
        color = PRIMITIVE_COLORS.get(seg.primitive, "#AAAAAA")
        label = seg.primitive.value.upper()
        q = round(ss.quality.composite, 2)
        # Only label segments wide enough to fit text; the rest rely on hover,
        # so thin slivers don't pile overlapping text on top of each other.
        wide_enough = seg.duration_frames() >= max(total_frames * 0.07, 1)
        fig.add_trace(go.Bar(
            x=[seg.duration_frames()],
            y=["Primitives"],
            base=[seg.start_idx],
            orientation="h",
            marker=dict(color=color, line=dict(color="white", width=1)),
            name=label,
            text=(label if wide_enough else ""),
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=10),
            hovertemplate=f"<b>{label}</b><br>Frames {seg.start_idx}–{seg.end_idx}<br>Quality: {q}<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Frame", range=[0, total_frames]),
        yaxis=dict(showticklabels=False),
        showlegend=False,
        height=120,
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def per_segment_bars(scored: list[ScoredSegment]) -> go.Figure:
    """Grouped bar chart of quality dimensions per segment."""
    # Drop UNKNOWN segments — they're noise and crowd the axis.
    scored = [ss for ss in scored if ss.segment.primitive != PrimitiveType.UNKNOWN]
    if not scored:
        return go.Figure()

    # Label each cluster by its frame range only (the timeline above already
    # names the primitives), keeping the axis readable.
    labels = [f"{ss.segment.start_idx}–{ss.segment.end_idx}" for ss in scored]
    metrics = {
        "Smoothness":      [ss.quality.smoothness for ss in scored],
        "Path Efficiency": [ss.quality.path_efficiency for ss in scored],
        "Decisiveness":    [ss.quality.decisiveness for ss in scored],
        "Confidence":      [ss.quality.confidence_mean for ss in scored],
    }
    colors = ["#4C9BE8", "#2A9D8F", "#E9C46A", "#F4A261"]

    fig = go.Figure()
    for (name, vals), color in zip(metrics.items(), colors):
        fig.add_trace(go.Bar(name=name, x=labels, y=vals, marker_color=color))

    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1], title="Score"),
        xaxis=dict(title="Frame range", tickangle=-45, type="category"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        bargap=0.25,
        height=340,
        margin=dict(l=10, r=10, t=40, b=70),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
