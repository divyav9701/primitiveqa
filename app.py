"""PrimitiveQA — Gradio web app (single + batch analysis)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import base64

import gradio as gr
import pandas as pd

import pipeline as pqa
from core.types import AnalysisResult, PrimitiveType, VLMEvaluation
from evaluation.vlm import (
    get_frames_b64, parse_response,
    prose_to_html, stream_chat, stream_prose,
)
from visualization.charts import per_segment_bars, primitive_timeline, quality_radar
from visualization.dataset_charts import (
    dataset_health_score,
    dataset_radar,
    missing_primitives_warning,
    primitive_coverage_chart,
    score_distribution,
    summary_table,
)


def _fig_to_html(fig, width: int = 460, height: int = 320) -> str:
    """Render a Plotly figure to a base64 PNG and return an <img> HTML tag.

    This avoids Gradio 6's HTML sanitiser stripping <script> tags (which
    breaks plotly.io.to_html CDN embeds) and the broken gr.Plot renderer.

    The image is capped at its natural ``width`` (and centered) so it never
    stretches to fill the full browser width.
    """
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
    b64 = base64.b64encode(img_bytes).decode()
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-width:{width}px;width:100%;border-radius:6px;'
        f'display:block;margin:0 auto">'
    )

EXAMPLES_DIR = Path(__file__).parent / "examples"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

PRIMITIVE_LABEL = {
    PrimitiveType.REACH:     "Reach",
    PrimitiveType.GRASP:     "Grasp",
    PrimitiveType.LIFT:      "Lift",
    PrimitiveType.TRANSPORT: "Transport",
    PrimitiveType.PLACE:     "Place",
    PrimitiveType.RETRACT:   "Retract",
    PrimitiveType.UNKNOWN:   "Unknown",
}


def _quality_color(score: float) -> str:
    if score >= 0.65:
        return "#2A9D8F"
    if score >= 0.4:
        return "#E9C46A"
    return "#E76F51"


# Hover-tooltip explanations, split into Definition / High / Low lines so each
# renders on its own row in the pop-up.
METRIC_INFO = {
    "Composite": {
        "def": "Overall quality = 0.35×Smoothness + 0.25×Path Efficiency + 0.25×Decisiveness + 0.15×Detection.",
        "note": "Above 0.75 = a clean pick-and-place.",
    },
    "Smoothness": {
        "def": "Jerkiness of motion.",
        "high": "gliding path",
        "low": "hesitation, mid-motion corrections",
    },
    "Path Efficiency": {
        "def": "Straight-line distance ÷ actual path taken.",
        "high": "reached straight for the object",
        "low": "wandering or curved",
    },
    "Decisiveness": {
        "def": "Speed consistency.",
        "high": "one steady pace",
        "low": "speeds up, slows, hesitates, pauses",
    },
    "Detection Conf": {
        "def": "Hand visibility — how often MediaPipe could find the hand.",
        "note": "Findable ≠ good movement.",
    },
}


def _tip_html(name: str) -> str:
    """Definition, High, and Low each on their own line for the pop-up tooltip."""
    d = METRIC_INFO.get(name, {})
    lines = [f"<b>{name}</b>", d.get("def", "")]
    if d.get("high"):
        lines.append(f'<b>High:</b> {d["high"]}')
    if d.get("low"):
        lines.append(f'<b>Low:</b> {d["low"]}')
    if d.get("note"):
        lines.append(f'<span style="opacity:.85">{d["note"]}</span>')
    return "<br>".join(line for line in lines if line)


APP_CSS = """
/* ── Reset / base ───────────────────────────────────────────── */
.gradio-container{max-width:1120px !important;margin:0 auto !important;
  padding:16px 20px !important}

/* ── Tabs ────────────────────────────────────────────────────── */
.tab-nav{border-bottom:2px solid #e5e7eb !important;margin-bottom:0 !important}
.tab-nav button{font-weight:600 !important;font-size:0.88rem !important;
  color:#6b7280 !important;padding:10px 22px !important;border:none !important;
  border-radius:0 !important;background:transparent !important}
.tab-nav button.selected{color:#4f46e5 !important;
  border-bottom:2px solid #4f46e5 !important}

/* ── Primary action buttons ─────────────────────────────────── */
#pqa-analyze-btn > button, #pqa-batch-btn > button{
  background:linear-gradient(135deg,#4338ca,#6366f1) !important;
  border:none !important;border-radius:10px !important;
  font-weight:700 !important;font-size:1rem !important;
  letter-spacing:0.01em !important;
  box-shadow:0 4px 16px rgba(99,102,241,0.38) !important;
  transition:box-shadow .18s,transform .18s !important}
#pqa-analyze-btn > button:hover, #pqa-batch-btn > button:hover{
  box-shadow:0 6px 22px rgba(99,102,241,0.5) !important;
  transform:translateY(-1px) !important}

/* ── Accordion (API key) ────────────────────────────────────── */
#pqa-api-key{border:1px solid #e5e7eb !important;border-radius:10px !important;
  background:#fafafa !important;margin-bottom:8px !important}

/* ── Section dividers ───────────────────────────────────────── */
.pqa-section{font-size:0.68rem;font-weight:700;text-transform:uppercase;
  letter-spacing:0.1em;color:#9ca3af;padding:14px 0 6px;
  border-top:1px solid #f3f4f6;margin-top:6px}

/* ── Hover tooltips + pill overflow ────────────────────────── */
.pqa-host,.pqa-host *{overflow:visible !important}
.pqa-pill{position:relative;display:inline-block;cursor:help;
  padding:4px 10px;margin:3px;border-radius:6px;
  font-size:0.84rem;border:1px solid #ccc}
.pqa-pill>.pqa-tip{visibility:hidden;opacity:0;position:absolute;
  top:140%;left:50%;transform:translateX(-50%);
  background:#1f2937;color:#fff;padding:8px 11px;border-radius:8px;
  width:230px;font-size:0.78rem;line-height:1.5;font-weight:400;
  z-index:9999;transition:opacity .12s;
  box-shadow:0 8px 24px rgba(0,0,0,.28);text-align:left;white-space:normal}
.pqa-pill>.pqa-tip,.pqa-pill>.pqa-tip *{color:#fff !important}
.pqa-pill>.pqa-tip::after{content:"";position:absolute;bottom:100%;left:50%;
  transform:translateX(-50%);border:6px solid transparent;
  border-bottom-color:#1f2937}
.pqa-pill:hover>.pqa-tip{visibility:visible;opacity:1}

/* ── Compare video players ──────────────────────────────────── */
.pqa-clip-video video,.pqa-clip-video{max-height:280px}
.pqa-clip-video video{object-fit:contain;border-radius:8px}

/* ── Dataset health score big number ───────────────────────── */
.pqa-health-score{text-align:center;padding:16px 0 4px}

/* ── Result video card ──────────────────────────────────────── */
#pqa-result-video{
  background:#0f172a !important;
  border:1px solid #1e3a5f !important;
  border-radius:12px !important;
  overflow:hidden !important;
  padding:0 !important}
#pqa-result-video .empty{background:#0f172a !important;border:none !important}
#pqa-result-video video{border-radius:0 !important;display:block;width:100%}
#pqa-result-video label{
  color:#334155 !important;font-size:0.66rem !important;
  font-weight:700 !important;text-transform:uppercase !important;
  letter-spacing:.1em !important;padding:10px 14px 6px !important}
"""

# ── Static HTML fragments ─────────────────────────────────────────────────────

HEADER_HTML = """
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
  border-radius:14px;padding:26px 32px;margin-bottom:16px">
  <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
    <div style="font-size:2.4rem;line-height:1;user-select:none">⬡</div>
    <div style="flex:1;min-width:200px">
      <div style="font-size:1.55rem;font-weight:800;color:#fff;
           letter-spacing:-0.02em;line-height:1.1">PrimitiveQA</div>
      <div style="font-size:0.82rem;color:#94a3b8;margin-top:4px">
        Quality layer for physical AI training data
      </div>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center">
      <span style="background:rgba(99,102,241,0.18);color:#a5b4fc;
        padding:4px 12px;border-radius:20px;font-size:0.72rem;font-weight:600;
        border:1px solid rgba(99,102,241,0.3)">→ reach</span>
      <span style="background:rgba(249,115,22,0.15);color:#fdba74;
        padding:4px 12px;border-radius:20px;font-size:0.72rem;font-weight:600;
        border:1px solid rgba(249,115,22,0.25)">✊ grasp</span>
      <span style="background:rgba(16,185,129,0.15);color:#6ee7b7;
        padding:4px 12px;border-radius:20px;font-size:0.72rem;font-weight:600;
        border:1px solid rgba(16,185,129,0.25)">↑ lift</span>
      <span style="background:rgba(234,179,8,0.15);color:#fde047;
        padding:4px 12px;border-radius:20px;font-size:0.72rem;font-weight:600;
        border:1px solid rgba(234,179,8,0.25)">⇒ transport</span>
      <span style="background:rgba(239,68,68,0.15);color:#fca5a5;
        padding:4px 12px;border-radius:20px;font-size:0.72rem;font-weight:600;
        border:1px solid rgba(239,68,68,0.25)">↓ place</span>
      <span style="background:rgba(139,92,246,0.15);color:#c4b5fd;
        padding:4px 12px;border-radius:20px;font-size:0.72rem;font-weight:600;
        border:1px solid rgba(139,92,246,0.25)">← retract</span>
    </div>
  </div>
</div>
"""


def _section(label: str) -> str:
    return f'<div class="pqa-section">{label}</div>'


def _metric_pill(name: str, value: float | None = None) -> str:
    """One hover-tooltip metric chip. With a value it's color-coded red/yellow/
    green; without one it's a neutral definition chip for the legend."""
    if value is None:
        label = f"{name} <span style='opacity:.45'>&#9432;</span>"
        bg, border = "transparent", "#cccccc"
    else:
        color = _quality_color(value)
        label = f"{name}: <b>{value:.2f}</b>"
        bg, border = f"{color}22", color
    return (
        f'<span class="pqa-pill" style="background:{bg};border-color:{border}">{label}'
        f'<span class="pqa-tip">{_tip_html(name)}</span></span>'
    )


def _metrics_panel(q) -> str:
    """Color-coded, tooltipped readout of the four sub-metrics for one clip."""
    pills = (
        _metric_pill("Smoothness", q.smoothness)
        + _metric_pill("Path Efficiency", q.path_efficiency)
        + _metric_pill("Decisiveness", q.decisiveness)
        + _metric_pill("Detection Conf", q.confidence_mean)
    )
    return f'<div style="text-align:center;margin-top:4px">{pills}</div>'


def _metrics_legend() -> str:
    """Static, always-visible legend — hover any metric to see what it measures."""
    pills = "".join(
        _metric_pill(n)
        for n in ["Composite", "Smoothness", "Path Efficiency", "Decisiveness", "Detection Conf"]
    )
    return (
        '<div style="text-align:center;font-size:0.9rem">'
        '<span style="opacity:.6">Hover a metric to see what it measures:</span><br>'
        f"{pills}</div>"
    )


def _sort_summary(col: str, df):
    """Re-sort the per-clip summary by a chosen column (Gradio's Dataframe has no
    native click-to-sort). Numeric columns sort high→low; text columns A→Z."""
    if df is None or not hasattr(df, "columns") or col not in df.columns:
        return gr.update()
    ascending = col in ("Clip", "Primitives Found")
    if col == "Detection":
        key = df["Detection"].map(
            lambda v: float(str(v).rstrip("%")) if str(v).rstrip("%").replace(".", "", 1).isdigit() else -1.0
        )
        ordered = df.assign(_k=key).sort_values("_k", ascending=ascending).drop(columns="_k")
    else:
        ordered = df.sort_values(col, ascending=ascending)
    return _style_summary(ordered)


def _style_summary(df):
    """Color-code the numeric quality cells red/yellow/green. Returns a pandas
    Styler — Gradio 6 carries these colors through ``metadata.styling`` while
    keeping the Dataframe's click-to-sort headers."""
    num_cols = [c for c in ["Composite", "Smoothness", "Path Eff.", "Decisive"] if c in df.columns]

    def shade_num(v):
        try:
            return f"background-color: {_quality_color(float(v))}33"
        except (TypeError, ValueError):
            return ""

    def shade_pct(v):
        try:
            return f"background-color: {_quality_color(float(str(v).rstrip('%')) / 100)}33"
        except (TypeError, ValueError):
            return ""

    sty = df.style.map(shade_num, subset=num_cols)
    if "Detection" in df.columns:
        sty = sty.map(shade_pct, subset=["Detection"])
    return sty


def _format_primitive_sequence(result: AnalysisResult) -> str:
    if not result.segments:
        return "No primitives detected."
    parts = []
    for ss in result.segments:
        if ss.segment.primitive == PrimitiveType.UNKNOWN:
            continue
        label = PRIMITIVE_LABEL.get(ss.segment.primitive, ss.segment.primitive.value)
        q = ss.quality.composite
        color = _quality_color(q)
        parts.append(
            f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;margin:2px;'
            f'font-size:0.85rem;background:{color}22;border:1px solid {color}">'
            f"{label} <b>{q:.2f}</b></span>"
        )
    return " ".join(parts) if parts else "<i>Only UNKNOWN segments detected — try a longer clip with clearer hand visibility.</i>"


_VLM_PRIM_ICON = {
    "reach": "→", "grasp": "✊", "lift": "↑",
    "transport": "⇒", "place": "↓", "retract": "←",
}
_VLM_QUALITY_COLOR = {"good": "#2A9D8F", "ok": "#E9C46A", "poor": "#E76F51"}


def _format_vlm(result: AnalysisResult) -> str:
    """Render Claude's video-reasoning output as a primitive timeline table."""
    v = result.vlm_eval
    if v is None or v.skipped:
        note = v.notes if v else "Not evaluated."
        return (
            f'<div style="color:#888;font-size:0.9rem;padding:8px 0">'
            f'<i>{note}</i></div>'
        )

    status_icon  = "✅" if v.task_success else "❌"
    status_color = "#2A9D8F" if v.task_success else "#E76F51"
    html = (
        f'<div style="font-weight:600;font-size:1rem;margin-bottom:10px">'
        f'{status_icon} <span style="color:{status_color}">{v.task_description}</span> '
        f'<span style="color:#888;font-weight:400;font-size:0.85rem">'
        f'({v.confidence:.0%} confidence)</span></div>'
    )

    if not v.segments:
        html += f'<i style="color:#888">{v.notes}</i>'
        return html

    # Timeline table
    html += (
        '<table style="width:100%;border-collapse:collapse;font-size:0.85rem">'
        '<tr style="color:#aaa;font-size:0.72rem;text-transform:uppercase;'
        'letter-spacing:.04em">'
        '<td style="padding:4px 10px 4px 4px">Time</td>'
        '<td style="padding:4px 10px">Primitive</td>'
        '<td style="padding:4px 10px">Quality</td>'
        '<td style="padding:4px 4px">Claude note</td>'
        '</tr>'
    )
    for seg in v.segments:
        icon  = _VLM_PRIM_ICON.get(seg.primitive, "?")
        color = _VLM_QUALITY_COLOR.get(seg.quality, "#888")
        badge = (
            f'<span style="background:{color}22;border:1px solid {color};'
            f'border-radius:4px;padding:1px 7px;color:{color};font-size:0.78rem">'
            f'{seg.quality}</span>'
        )
        html += (
            f'<tr style="border-top:1px solid #eee">'
            f'<td style="color:#888;white-space:nowrap;padding:6px 10px 6px 4px">'
            f'{seg.start_sec:.1f}–{seg.end_sec:.1f}s</td>'
            f'<td style="font-weight:600;padding:6px 10px">{icon} {seg.primitive.capitalize()}</td>'
            f'<td style="padding:6px 10px">{badge}</td>'
            f'<td style="padding:6px 4px;color:#555">{seg.note}</td>'
            f'</tr>'
        )
    html += '</table>'

    if v.notes:
        html += (
            f'<div style="margin-top:8px;color:#888;font-size:0.82rem">'
            f'<i>{v.notes}</i></div>'
        )
    return html


def _export_dataset_json(results: list[AnalysisResult], names: list[str]) -> Path:
    data = {
        "dataset_health_score": round(dataset_health_score(results), 3),
        "n_clips": len(results),
        "clips": [
            {
                "name": name,
                "overall_quality": r.overall_quality.to_dict(),
                "segments": [
                    {
                        "primitive": ss.segment.primitive.value,
                        "start_frame": ss.segment.start_idx,
                        "end_frame": ss.segment.end_idx,
                        "quality": ss.quality.to_dict(),
                    }
                    for ss in r.segments
                ],
                "trajectory_length": len(r.trajectory),
                "detection_rate": round(
                    sum(1 for p in r.trajectory.points if p.confidence > 0)
                    / max(len(r.trajectory.points), 1), 3
                ),
            }
            for name, r in zip(names, results)
        ],
    }
    out = OUTPUT_DIR / "dataset_health.json"
    out.write_text(json.dumps(data, indent=2))
    return out


def _render_chat(history: list[dict], streaming: bool = False) -> str:
    """Render conversation history as styled chat bubbles."""
    import html as _html

    if not history:
        return (
            '<div style="background:#0f172a;border-radius:10px;padding:28px;'
            'text-align:center;color:#475569;font-size:0.83rem;min-height:120px;'
            'display:flex;align-items:center;justify-content:center">'
            'Ask Claude anything about the video — movement quality, specific moments, '
            'what primitives you saw, whether the data is usable…'
            '</div>'
        )

    def _md(text: str) -> str:
        s = _html.escape(text)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"\*(.+?)\*", r'<i style="color:#94a3b8">\1</i>', s)
        return s.replace("\n", "<br>")

    html_parts = [
        '<div style="background:#0f172a;border-radius:10px;padding:16px;'
        'display:flex;flex-direction:column;gap:10px">'
    ]
    for i, turn in enumerate(history):
        is_last = i == len(history) - 1
        if turn["role"] == "user":
            html_parts.append(
                '<div style="display:flex;justify-content:flex-end">'
                f'<div style="background:#312e81;color:#e0e7ff;padding:10px 14px;'
                f'border-radius:10px 10px 2px 10px;max-width:82%;font-size:0.84rem;'
                f'line-height:1.5">{_html.escape(turn["content"])}</div></div>'
            )
        else:
            cursor = '<span style="color:#6366f1">▌</span>' if streaming and is_last else ""
            html_parts.append(
                '<div style="display:flex;justify-content:flex-start">'
                f'<div style="background:#1e293b;color:#cbd5e1;padding:10px 14px;'
                f'border-radius:10px 10px 10px 2px;max-width:90%;font-size:0.84rem;'
                f'line-height:1.6">{_md(turn["content"])}{cursor}</div></div>'
            )
    html_parts.append("</div>")
    return "".join(html_parts)


# ── Single-video analysis ────────────────────────────────────────────────────

_BLANK9 = (None, "Skeleton overlay", None, "", "", "", "", "", "")


def analyze_single(video_path: str | None, api_key: str):
    """Generator — yields partial UI updates so Claude streams live."""
    if not video_path:
        raise gr.Error("Please upload a video first.")

    def _dark_status(msg: str) -> str:
        return (
            '<div style="background:#0f172a;border-radius:10px;padding:20px;'
            'min-height:400px;display:flex;align-items:center;justify-content:center">'
            f'<span style="color:#94a3b8;font-size:0.82rem">{msg}</span></div>'
        )

    # ── Phase 1: show "pipeline running" while MediaPipe tracks ─────────────
    yield (
        None, "Skeleton overlay", None,
        '<div style="color:#888;text-align:center">⏳ Tracking hand…</div>',
        "", "", "", "",
        _dark_status("⏳ Tracking hand with MediaPipe…"),
        [],   # frames_state — empty until pipeline done
    )

    try:
        result = pqa.run(video_path, api_key=None, skip_vlm=True)
    except Exception as e:
        raise gr.Error(f"Pipeline error: {e}")

    # ── Phase 2: pipeline done — build score, copy videos, start charts ─────
    q = result.overall_quality
    color = _quality_color(q.composite)
    score_html = (
        f'<div style="font-size:2rem;font-weight:bold;text-align:center;color:{color}">'
        f"Composite: {q.composite:.0%}</div>"
        + _metrics_panel(q)
    )
    annotated = str(Path(tempfile.mktemp(suffix=".mp4")))
    shutil.copy2(Path(result.annotated_video_path).resolve(), annotated)
    original = str(Path(tempfile.mktemp(suffix=".mp4")))
    shutil.copy2(Path(video_path).resolve(), original)
    video_paths = {"Skeleton overlay": annotated, "Original": original}

    key = api_key.strip() if api_key else ""
    vlm_placeholder = (
        _dark_status("No API key set — add one above to enable Claude reasoning.")
        if not key else
        _dark_status("⏳ Asking Claude…")
    )
    frames_b64 = get_frames_b64(video_path) if key else []
    prim_html  = _format_primitive_sequence(result)

    # Shimmer skeleton shown while each chart is still rendering
    def _shimmer(h: int = 220) -> str:
        return (
            f'<div style="background:linear-gradient(90deg,#1e293b 25%,#273548 50%,'
            f'#1e293b 75%);background-size:200% 100%;border-radius:8px;height:{h}px;'
            f'animation:pqa-shimmer 1.4s ease-in-out infinite"></div>'
        )

    # ── Phase 2a: show video + score; charts shimmer ─────────────────────────
    yield (
        annotated, "Skeleton overlay", video_paths,
        score_html, _shimmer(280), _shimmer(200), _shimmer(300), "",
        vlm_placeholder, frames_b64,
    )

    # ── Phase 2b: render + reveal charts one at a time ───────────────────────
    import time

    time.sleep(1.0)
    radar_html = _fig_to_html(quality_radar(result), width=380, height=280)
    yield (
        annotated, "Skeleton overlay", video_paths,
        score_html, radar_html, _shimmer(200), _shimmer(300), "",
        vlm_placeholder, frames_b64,
    )

    time.sleep(1.2)
    timeline_html = _fig_to_html(
        primitive_timeline(result.segments, len(result.trajectory)), width=380, height=200
    )
    yield (
        annotated, "Skeleton overlay", video_paths,
        score_html, radar_html, timeline_html, _shimmer(300), "",
        vlm_placeholder, frames_b64,
    )

    time.sleep(1.2)
    bars_html = _fig_to_html(per_segment_bars(result.segments), width=380, height=300)
    yield (
        annotated, "Skeleton overlay", video_paths,
        score_html, radar_html, timeline_html, bars_html, prim_html,
        vlm_placeholder, frames_b64,
    )

    if not key:
        return

    # ── Phase 3: stream Claude token by token into the panel ─────────────────
    accumulated = ""
    # ── Phase 3: stream Claude prose token by token ───────────────────────────
    for chunk in stream_prose(video_path, api_key=key):
        accumulated += chunk
        yield (
            annotated, "Skeleton overlay", video_paths,
            score_html, radar_html, timeline_html, bars_html, prim_html,
            prose_to_html(accumulated, cursor=True),
            frames_b64,
        )

    # ── Phase 4: finalise — remove cursor ─────────────────────────────────────
    final_vlm = (
        prose_to_html(accumulated, cursor=False)
        if accumulated else
        _dark_status("Claude returned no output.")
    )
    yield (
        annotated, "Skeleton overlay", video_paths,
        score_html, radar_html, timeline_html, bars_html, prim_html,
        final_vlm,
        frames_b64,
    )


def chat_about_video(message: str, history: list, frames_b64: list, api_key: str):
    """Generator: stream Claude's reply and update the chat display."""
    if not message.strip():
        yield history, _render_chat(history), ""
        return

    if not frames_b64:
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Analyze a video first, then ask me anything about it."},
        ]
        yield new_history, _render_chat(new_history), ""
        return

    key = api_key.strip() if api_key else ""
    if not key:
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Add an Anthropic API key above to enable chat."},
        ]
        yield new_history, _render_chat(new_history), ""
        return

    # Show user message immediately
    history = history + [{"role": "user", "content": message}]
    yield history, _render_chat(history), ""

    # Stream Claude's response
    accumulated = ""
    for chunk in stream_chat(frames_b64, history[:-1], message, api_key=key):
        accumulated += chunk
        streaming = history + [{"role": "assistant", "content": accumulated}]
        yield streaming, _render_chat(streaming, streaming=True), ""

    final = history + [{"role": "assistant", "content": accumulated or "…"}]
    yield final, _render_chat(final, streaming=False), ""


def _swap_video(choice: str, paths: dict | None):
    """Switch the result player between the skeleton overlay and the original."""
    if not paths:
        return gr.update()
    return paths.get(choice)


def _clip_info_html(name: str, r: AnalysisResult) -> str:
    """Composite score + metric pills + Claude task evaluation for one clip."""
    q = r.overall_quality
    color = _quality_color(q.composite)
    return (
        f'<div style="font-size:1.3rem;font-weight:bold;text-align:center;color:{color}">'
        f"{name}: {q.composite:.0%}</div>"
        + _metrics_panel(q)
        + '<div style="margin-top:6px;text-align:center"><b>Claude video reasoning</b></div>'
        + f'<div style="text-align:center">{_format_vlm(r)}</div>'
    )


def _clip_video(clip: str | None, version: str, data: dict | None):
    """Show the chosen clip (overlay or original) in one compare player."""
    if not data or clip not in (data or {}):
        return gr.update()
    return data[clip]["videos"].get(version)


def _clip_select(clip: str | None, version: str, data: dict | None):
    """When a clip is picked, refresh both its player and its info panel."""
    if not data or clip not in (data or {}):
        return gr.update(), gr.update()
    return data[clip]["videos"].get(version), data[clip]["info"]


# ── Batch / dataset analysis ─────────────────────────────────────────────────

def analyze_batch(video_files: list | None, api_key: str, progress=gr.Progress()) -> tuple:
    if not video_files:
        raise gr.Error("Upload at least one video.")

    paths = [v if isinstance(v, str) else v.name for v in video_files]
    names = [Path(p).name for p in paths]
    results: list[AnalysisResult] = []

    for i, (path, name) in enumerate(zip(paths, names)):
        progress((i / len(paths)), desc=f"Processing {name} ({i+1}/{len(paths)})…")
        try:
            results.append(pqa.run(path, api_key=api_key or None))
        except Exception as e:
            raise gr.Error(f"Failed on {name}: {e}")

    progress(1.0, desc="Done!")

    health = dataset_health_score(results)
    color = _quality_color(health)
    health_html = (
        f'<div style="font-size:2.2rem;font-weight:bold;text-align:center;color:{color}">'
        f"Dataset Health: {health:.0%}</div>"
        f'<div style="text-align:center;color:#888;font-size:0.9rem">'
        f"{len(results)} clips analyzed</div>"
    )

    df = summary_table(results, names).sort_values("Composite", ascending=False)
    export_path = _export_dataset_json(results, names)

    # Copy each clip's overlay + original into temp and pre-render its info
    # panel (scores + Claude eval) so the compare view can pull any clip up.
    clip_data: dict[str, dict] = {}
    for name, path, r in zip(names, paths, results):
        overlay = str(Path(tempfile.mktemp(suffix=".mp4")))
        shutil.copy2(Path(r.annotated_video_path).resolve(), overlay)
        original = str(Path(tempfile.mktemp(suffix=".mp4")))
        shutil.copy2(Path(path).resolve(), original)
        clip_data[name] = {
            "videos": {"Skeleton overlay": overlay, "Original": original},
            "info": _clip_info_html(name, r),
        }

    clip_a = names[0] if names else None
    clip_b = names[1] if len(names) > 1 else clip_a

    def _initial(clip):
        if not clip:
            return None, ""
        return clip_data[clip]["videos"]["Skeleton overlay"], clip_data[clip]["info"]

    a_video, a_info = _initial(clip_a)
    b_video, b_info = _initial(clip_b)

    return (
        health_html,
        missing_primitives_warning(results),
        _fig_to_html(dataset_radar(results)),
        _fig_to_html(primitive_coverage_chart(results), width=440, height=340),
        _fig_to_html(score_distribution(results, names), width=440, height=340),
        _style_summary(df),
        df,                   # plain frame stashed for the Sort-by control
        "Composite",          # reset the Sort-by dropdown
        clip_data,            # per-clip videos + info for the compare view
        gr.update(choices=names, value=clip_a),  # Clip A picker
        "Skeleton overlay",   # Clip A toggle
        a_video, a_info,      # Clip A player + info
        gr.update(choices=names, value=clip_b),  # Clip B picker
        "Skeleton overlay",   # Clip B toggle
        b_video, b_info,      # Clip B player + info
        str(export_path),
    )


# ── UI ───────────────────────────────────────────────────────────────────────

AUTOPLAY_JS = """
() => {
  const RATE = 0.4;

  // Gradio 6's Svelte video component keeps a reactive signal initialised
  // to 1 and writes video.playbackRate = 1 on every re-render, overriding
  // a plain assignment. Fix: intercept the property on the element itself
  // via Object.defineProperty so every future write is ignored and the
  // native setter always receives our rate.
  function lockRate(v) {
    if (!v || v._pqaLocked) return;
    v._pqaLocked = true;
    const native = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, 'playbackRate');
    Object.defineProperty(v, 'playbackRate', {
      configurable: true,
      enumerable:   true,
      get: ()    => RATE,
      set: ()    => native.set.call(v, RATE),   // ignore caller's value
    });
    native.set.call(v, RATE);   // apply immediately
  }

  setInterval(() => {

    // ── 1. Slow-loop the result video ─────────────────────────────────────
    const vc = document.getElementById('pqa-result-video');
    if (vc) {
      const v = vc.querySelector('video');
      if (v) {
        lockRate(v);
        v.muted = true;
        v.loop  = true;
        if (v.src && v.paused && v.readyState >= 2) v.play().catch(() => {});
      }
    }

    // ── 2. Auto-scroll both the Claude prose panel and the chat display ───
    ['pqa-vlm-display', 'pqa-chat-display'].forEach(id => {
      const host = document.getElementById(id);
      if (!host) return;
      host.scrollTop = host.scrollHeight;
      host.querySelectorAll('div').forEach(d => {
        if (d.scrollHeight > d.clientHeight + 4) d.scrollTop = d.scrollHeight;
      });
    });

  }, 250);
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="PrimitiveQA") as demo:

        # ── Global style injection (tooltips need <style> tag) ────────────
        gr.HTML(
            '<style>'
            '.pqa-host,.pqa-host *{overflow:visible !important}'
            '.pqa-pill{position:relative;display:inline-block;cursor:help;'
            '  padding:4px 10px;margin:3px;border-radius:6px;font-size:.84rem;border:1px solid #ccc}'
            '.pqa-pill>.pqa-tip{visibility:hidden;opacity:0;position:absolute;'
            '  top:140%;left:50%;transform:translateX(-50%);background:#1f2937;color:#fff;'
            '  padding:8px 11px;border-radius:8px;width:230px;font-size:.78rem;line-height:1.5;'
            '  font-weight:400;z-index:9999;transition:opacity .12s;'
            '  box-shadow:0 8px 24px rgba(0,0,0,.28);text-align:left;white-space:normal}'
            '.pqa-pill>.pqa-tip,.pqa-pill>.pqa-tip *{color:#fff !important}'
            '.pqa-pill>.pqa-tip::after{content:"";position:absolute;bottom:100%;left:50%;'
            '  transform:translateX(-50%);border:6px solid transparent;border-bottom-color:#1f2937}'
            '.pqa-pill:hover>.pqa-tip{visibility:visible;opacity:1}'
            '.pqa-clip-video video,.pqa-clip-video{max-height:280px}'
            '.pqa-clip-video video{object-fit:contain;border-radius:8px}'
            '@keyframes pqa-shimmer{'
            '0%{background-position:200% 0}'
            '100%{background-position:-200% 0}}'
            '</style>',
            container=False,
        )

        gr.HTML(HEADER_HTML, container=False)

        with gr.Accordion("🔑  Anthropic API key", open=False, elem_id="pqa-api-key"):
            api_key_box = gr.Textbox(
                label="",
                placeholder="sk-ant-...  (optional — enables Claude video reasoning, ~$0.05/clip)",
                type="password",
                value=os.environ.get("ANTHROPIC_API_KEY", ""),
                show_label=False,
            )

        with gr.Tabs():

            # ══ Tab 1: Single video ══════════════════════════════════════════
            with gr.Tab("  Single video  "):

                # ── Row 1: upload ────────────────────────────────────────────
                gr.HTML(_section("📥  Input"))
                with gr.Row():
                    with gr.Column(scale=3):
                        video_in = gr.Video(
                            label="Upload a manipulation video",
                            sources=["upload"],
                        )
                    with gr.Column(scale=1, min_width=140):
                        gr.HTML("<div style='height:8px'></div>", container=False)
                        run_btn = gr.Button(
                            "▶  Analyze", variant="primary", size="lg",
                            elem_id="pqa-analyze-btn",
                        )

                # ── Three-column live view: Charts | Video | Claude + Chat ───
                gr.HTML(_section("📹  Results"))
                frames_state       = gr.State(value=[])
                chat_history_state = gr.State(value=[])
                video_paths_state  = gr.State()

                with gr.Row(equal_height=False):

                    # ── Left: charts populate as analysis runs ───────────────
                    with gr.Column(scale=1):
                        gr.HTML(
                            '<div style="font-size:0.68rem;font-weight:700;text-transform:'
                            'uppercase;letter-spacing:.1em;color:#9ca3af;margin-bottom:6px">'
                            '📈  Motion analysis</div>',
                            container=False,
                        )
                        score_display = gr.HTML(elem_classes=["pqa-host"])
                        radar_plot    = gr.HTML()
                        timeline_plot = gr.HTML()
                        bars_plot     = gr.HTML()
                        gr.HTML(
                            '<div style="font-size:0.68rem;font-weight:700;text-transform:'
                            'uppercase;letter-spacing:.1em;color:#9ca3af;margin:8px 0 4px">'
                            '🏷  Primitives detected</div>',
                            container=False,
                        )
                        prim_display = gr.HTML()

                    # ── Middle: result video ─────────────────────────────────
                    with gr.Column(scale=1):
                        video_toggle = gr.Radio(
                            ["Skeleton overlay", "Original"],
                            value="Skeleton overlay",
                            label="Show",
                        )
                        video_out = gr.Video(label="", elem_id="pqa-result-video")

                    # ── Right: Claude reasoning + chat ───────────────────────
                    with gr.Column(scale=1):
                        gr.HTML(
                            '<div style="display:flex;align-items:center;gap:8px;'
                            'margin-bottom:6px">'
                            '<span style="font-size:0.68rem;font-weight:700;'
                            'text-transform:uppercase;letter-spacing:.1em;color:#9ca3af">'
                            '🤖  Claude reasoning</span>'
                            '<span style="font-size:0.72rem;color:#c4b5fd;'
                            'background:rgba(139,92,246,0.12);border:1px solid '
                            'rgba(139,92,246,0.25);border-radius:12px;padding:1px 8px">'
                            'streams live</span></div>',
                            container=False,
                        )
                        vlm_display = gr.HTML(
                            '<div style="background:#0f172a;border-radius:10px;'
                            'padding:20px;min-height:280px;display:flex;'
                            'align-items:center;justify-content:center">'
                            '<span style="color:#475569;font-size:0.82rem">'
                            'Click Analyze — reasoning streams here live.'
                            '</span></div>',
                            elem_id="pqa-vlm-display",
                        )
                        gr.HTML(
                            '<div style="font-size:0.68rem;font-weight:700;text-transform:'
                            'uppercase;letter-spacing:.1em;color:#9ca3af;margin:10px 0 6px">'
                            '💬  Ask Claude</div>',
                            container=False,
                        )
                        chat_display = gr.HTML(
                            _render_chat([]),
                            elem_id="pqa-chat-display",
                        )
                        with gr.Row():
                            chat_input = gr.Textbox(
                                placeholder="Ask anything about the video…",
                                show_label=False,
                                scale=4,
                                container=False,
                            )
                            send_btn = gr.Button("↑", variant="primary", scale=1, min_width=48)

                gr.HTML(_metrics_legend(), elem_classes=["pqa-host"])

                run_btn.click(
                    fn=analyze_single,
                    inputs=[video_in, api_key_box],
                    outputs=[video_out, video_toggle, video_paths_state,
                             score_display, radar_plot, timeline_plot,
                             bars_plot, prim_display, vlm_display, frames_state],
                )
                video_toggle.change(
                    fn=_swap_video,
                    inputs=[video_toggle, video_paths_state],
                    outputs=video_out,
                )
                chat_inputs  = [chat_input, chat_history_state, frames_state, api_key_box]
                chat_outputs = [chat_history_state, chat_display, chat_input]
                send_btn.click(fn=chat_about_video, inputs=chat_inputs, outputs=chat_outputs)
                chat_input.submit(fn=chat_about_video, inputs=chat_inputs, outputs=chat_outputs)

            # ══ Tab 2: Dataset health ════════════════════════════════════════
            with gr.Tab("  Dataset health  "):
                gr.HTML(
                    '<div style="font-size:0.85rem;color:#6b7280;padding:4px 0 12px">'
                    'Upload multiple clips to get a dataset-level quality report — '
                    'primitive coverage, score distribution, and overall health score.'
                    '</div>',
                    container=False,
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML(_section("📥  Upload clips"))
                        batch_files = gr.File(
                            label="",
                            file_count="multiple",
                            file_types=["video"],
                            show_label=False,
                        )
                        batch_btn = gr.Button(
                            "Analyze dataset", variant="primary", size="lg",
                            elem_id="pqa-batch-btn",
                        )
                        gr.HTML(_section("⚠️  Coverage gaps"))
                        missing_display = gr.HTML()

                    with gr.Column(scale=1):
                        gr.HTML(_section("🏥  Dataset health score"))
                        health_display = gr.HTML(elem_classes=["pqa-health-score"])
                        dataset_radar_plot = gr.HTML()

                gr.HTML(_metrics_legend(), elem_classes=["pqa-host"])

                gr.HTML(_section("📊  Coverage & score distribution"))
                with gr.Row(equal_height=True):
                    coverage_plot = gr.HTML()
                    dist_plot     = gr.HTML()

                # ── Compare view ─────────────────────────────────────────────
                gr.HTML(_section("🔍  Preview & compare clips"))
                clip_data_state = gr.State()
                with gr.Row():
                    with gr.Column(scale=1):
                        clip_a_picker = gr.Dropdown(choices=[], label="Clip A")
                        clip_a_toggle = gr.Radio(
                            ["Skeleton overlay", "Original"],
                            value="Skeleton overlay", label="Show",
                        )
                        clip_a_video = gr.Video(label="", height=280, elem_classes=["pqa-clip-video"])
                        clip_a_info = gr.HTML(elem_classes=["pqa-host"])
                    with gr.Column(scale=1):
                        clip_b_picker = gr.Dropdown(choices=[], label="Clip B")
                        clip_b_toggle = gr.Radio(
                            ["Skeleton overlay", "Original"],
                            value="Skeleton overlay", label="Show",
                        )
                        clip_b_video = gr.Video(label="", height=280, elem_classes=["pqa-clip-video"])
                        clip_b_info = gr.HTML(elem_classes=["pqa-host"])

                gr.HTML(_section("📋  Per-clip summary"))
                with gr.Row():
                    summary_sort = gr.Dropdown(
                        ["Composite", "Clip", "Detection", "Smoothness", "Path Eff.", "Decisive"],
                        value="Composite",
                        label="Sort by",
                        scale=1,
                    )
                summary_df    = gr.Dataframe(label="", wrap=True, show_label=False)
                summary_state = gr.State()
                export_file   = gr.File(label="Download dataset_health.json")

                batch_btn.click(
                    fn=analyze_batch,
                    inputs=[batch_files, api_key_box],
                    outputs=[
                        health_display,
                        missing_display,
                        dataset_radar_plot,
                        coverage_plot,
                        dist_plot,
                        summary_df,
                        summary_state,
                        summary_sort,
                        clip_data_state,
                        clip_a_picker, clip_a_toggle, clip_a_video, clip_a_info,
                        clip_b_picker, clip_b_toggle, clip_b_video, clip_b_info,
                        export_file,
                    ],
                )
                summary_sort.change(
                    fn=_sort_summary,
                    inputs=[summary_sort, summary_state],
                    outputs=summary_df,
                )
                # Each column: picking a clip refreshes its player + info; the
                # toggle just swaps overlay/original in that player.
                clip_a_picker.change(
                    fn=_clip_select,
                    inputs=[clip_a_picker, clip_a_toggle, clip_data_state],
                    outputs=[clip_a_video, clip_a_info],
                )
                clip_a_toggle.change(
                    fn=_clip_video,
                    inputs=[clip_a_picker, clip_a_toggle, clip_data_state],
                    outputs=clip_a_video,
                )
                clip_b_picker.change(
                    fn=_clip_select,
                    inputs=[clip_b_picker, clip_b_toggle, clip_data_state],
                    outputs=[clip_b_video, clip_b_info],
                )
                clip_b_toggle.change(
                    fn=_clip_video,
                    inputs=[clip_b_picker, clip_b_toggle, clip_data_state],
                    outputs=clip_b_video,
                )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue()
    app.launch(
        share=False,
        show_error=True,
        allowed_paths=[tempfile.gettempdir()],
        css=APP_CSS,
        js=AUTOPLAY_JS,
    )
