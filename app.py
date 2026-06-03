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
from evaluation.vlm import parse_response, stream_raw
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


# CSS for the pop-up hover tooltips (native `title` tooltips are tiny and don't
# work on chart images, so we draw our own). Injected once via gr.HTML; a <style>
# block applies page-wide. The tooltip sits below the pill and the legend/panel
# reserve padding-bottom so it's never clipped by the component box.
TOOLTIP_CSS = """
<style>
/* let the pop-up escape its component box instead of being clipped */
.pqa-host, .pqa-host *{overflow:visible !important}
.pqa-pill{position:relative;display:inline-block;cursor:help;padding:4px 10px;
  margin:3px;border-radius:6px;font-size:0.85rem;border:1px solid #ccc}
.pqa-pill > .pqa-tip{visibility:hidden;opacity:0;position:absolute;top:140%;
  left:50%;transform:translateX(-50%);background:#1f2937;color:#fff;padding:8px 11px;
  border-radius:6px;width:230px;font-size:0.78rem;line-height:1.5;font-weight:400;
  z-index:9999;transition:opacity .12s;box-shadow:0 6px 18px rgba(0,0,0,.28);
  text-align:left;white-space:normal}
/* Gradio's theme darkens <b>/headings; force all tooltip text white */
.pqa-pill > .pqa-tip, .pqa-pill > .pqa-tip *{color:#fff !important}
.pqa-pill > .pqa-tip::after{content:"";position:absolute;bottom:100%;left:50%;
  transform:translateX(-50%);border:6px solid transparent;border-bottom-color:#1f2937}
.pqa-pill:hover > .pqa-tip{visibility:visible;opacity:1}
/* keep both compare players short enough to sit side by side on one screen */
.pqa-clip-video video, .pqa-clip-video{max-height:300px}
.pqa-clip-video video{object-fit:contain}
</style>
"""


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


# ── Single-video analysis ────────────────────────────────────────────────────

_BLANK9 = (None, "Skeleton overlay", None, "", "", "", "", "", "")


def analyze_single(video_path: str | None, api_key: str):
    """Generator — yields partial UI updates so Claude streams live."""
    if not video_path:
        raise gr.Error("Please upload a video first.")

    # ── Phase 1: show "pipeline running" while MediaPipe tracks ─────────────
    yield (
        None, "Skeleton overlay", None,
        '<div style="color:#888;text-align:center">⏳ Tracking hand…</div>',
        "", "", "", "",
        '<div style="color:#888;font-size:0.9rem"><i>Waiting for pipeline…</i></div>',
    )

    try:
        result = pqa.run(video_path, api_key=None, skip_vlm=True)
    except Exception as e:
        raise gr.Error(f"Pipeline error: {e}")

    # ── Phase 2: pipeline done — show all charts; start Claude streaming ─────
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

    radar_html    = _fig_to_html(quality_radar(result))
    timeline_html = _fig_to_html(primitive_timeline(result.segments, len(result.trajectory)), width=520, height=240)
    bars_html     = _fig_to_html(per_segment_bars(result.segments), width=900, height=380)
    prim_html     = _format_primitive_sequence(result)

    key = api_key.strip() if api_key else ""
    vlm_placeholder = (
        '<div style="color:#888;font-size:0.9rem"><i>No API key — skipping Claude reasoning.</i></div>'
        if not key else
        '<div style="color:#888;font-size:0.9rem"><i>⏳ Asking Claude…</i></div>'
    )

    yield (
        annotated, "Skeleton overlay", video_paths,
        score_html, radar_html, timeline_html, bars_html, prim_html,
        vlm_placeholder,
    )

    if not key:
        return

    # ── Phase 3: stream Claude token by token into the panel ─────────────────
    accumulated = ""
    stream_html_wrap = (
        lambda t: (
            '<div style="font-family:monospace;font-size:0.78rem;'
            'background:#f8f8f8;border-radius:6px;padding:10px;'
            'max-height:280px;overflow-y:auto;white-space:pre-wrap;'
            'color:#333">' + t.replace("<", "&lt;").replace(">", "&gt;") + "▌</div>"
        )
    )
    for chunk in stream_raw(video_path, api_key=key):
        accumulated += chunk
        yield (
            annotated, "Skeleton overlay", video_paths,
            score_html, radar_html, timeline_html, bars_html, prim_html,
            stream_html_wrap(accumulated),
        )

    # ── Phase 4: parse and render final table ─────────────────────────────────
    if accumulated:
        vlm_eval = parse_response(accumulated)
        result.vlm_eval = vlm_eval
        final_vlm = _format_vlm(result)
    else:
        final_vlm = '<div style="color:#888;font-size:0.9rem"><i>Claude returned no output.</i></div>'

    yield (
        annotated, "Skeleton overlay", video_paths,
        score_html, radar_html, timeline_html, bars_html, prim_html,
        final_vlm,
    )


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

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="PrimitiveQA") as demo:
        gr.HTML(TOOLTIP_CSS, container=False)
        gr.Markdown(
            "# PrimitiveQA\n"
            "**The quality layer for physical AI data.** "
            "Decompose manipulation videos into skill primitives and score their quality."
        )

        api_key_box = gr.Textbox(
            label="Anthropic API key (optional — enables Claude task evaluation)",
            placeholder="sk-ant-...",
            type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
        )

        with gr.Tabs():

            # ── Tab 1: Single video ──────────────────────────────────────────
            with gr.Tab("Single video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_in = gr.Video(label="Upload manipulation video", sources=["upload"])
                        run_btn = gr.Button("Analyze", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        video_toggle = gr.Radio(
                            ["Skeleton overlay", "Original"],
                            value="Skeleton overlay",
                            label="Show in player",
                            info="After analyzing, switch the player between the tracked-skeleton overlay and your original clip.",
                        )
                        video_out = gr.Video(label="Result")
                        video_paths_state = gr.State()
                        score_display = gr.HTML(elem_classes=["pqa-host"])

                gr.HTML(_metrics_legend(), elem_classes=["pqa-host"])

                with gr.Row():
                    radar_plot   = gr.HTML(label="Quality radar")
                    timeline_plot = gr.HTML(label="Primitive timeline")

                bars_plot   = gr.HTML(label="Per-segment quality breakdown")
                prim_display = gr.HTML(label="Detected primitive sequence")
                gr.Markdown(
                    "### Claude video reasoning\n"
                    "*Requires API key — Claude watches the video and labels each primitive "
                    "with timestamps and a quality note.*"
                )
                vlm_display = gr.HTML()

                run_btn.click(
                    fn=analyze_single,
                    inputs=[video_in, api_key_box],
                    outputs=[video_out, video_toggle, video_paths_state, score_display,
                             radar_plot, timeline_plot, bars_plot, prim_display, vlm_display],
                )
                video_toggle.change(
                    fn=_swap_video,
                    inputs=[video_toggle, video_paths_state],
                    outputs=video_out,
                )

            # ── Tab 2: Dataset health ────────────────────────────────────────
            with gr.Tab("Dataset health"):
                gr.Markdown(
                    "Upload multiple clips to get a dataset-level quality report: "
                    "primitive coverage, score distribution, and overall health score."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        batch_files = gr.File(
                            label="Upload video clips",
                            file_count="multiple",
                            file_types=["video"],
                        )
                        batch_btn = gr.Button("Analyze dataset", variant="primary", size="lg")
                        # Primitive-coverage status sits right beneath Analyze.
                        missing_display = gr.HTML()

                    # Health score + the round radar group on the right.
                    with gr.Column(scale=1):
                        health_display = gr.HTML()
                        dataset_radar_plot = gr.HTML(label="Average quality radar")

                gr.HTML(_metrics_legend(), elem_classes=["pqa-host"])

                # The two bar charts side by side, same size.
                with gr.Row(equal_height=True):
                    coverage_plot = gr.HTML(label="Primitive coverage")
                    dist_plot     = gr.HTML(label="Per-clip composite scores")

                # Preview & compare two clips side by side.
                gr.Markdown("### Preview & compare clips")
                clip_data_state = gr.State()
                with gr.Row():
                    with gr.Column(scale=1):
                        clip_a_picker = gr.Dropdown(choices=[], label="Clip A")
                        clip_a_toggle = gr.Radio(
                            ["Skeleton overlay", "Original"],
                            value="Skeleton overlay", label="Show in player",
                        )
                        clip_a_video = gr.Video(label="Clip A preview", height=300, elem_classes=["pqa-clip-video"])
                        clip_a_info = gr.HTML(elem_classes=["pqa-host"])
                    with gr.Column(scale=1):
                        clip_b_picker = gr.Dropdown(choices=[], label="Clip B")
                        clip_b_toggle = gr.Radio(
                            ["Skeleton overlay", "Original"],
                            value="Skeleton overlay", label="Show in player",
                        )
                        clip_b_video = gr.Video(label="Clip B preview", height=300, elem_classes=["pqa-clip-video"])
                        clip_b_info = gr.HTML(elem_classes=["pqa-host"])

                with gr.Row():
                    summary_sort = gr.Dropdown(
                        ["Composite", "Clip", "Detection", "Smoothness", "Path Eff.", "Decisive"],
                        value="Composite",
                        label="Sort by",
                        scale=1,
                    )
                summary_df    = gr.Dataframe(label="Per-clip summary", wrap=True)
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
    )
