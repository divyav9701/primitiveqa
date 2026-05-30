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
from core.types import AnalysisResult, PrimitiveType
from visualization.charts import per_segment_bars, primitive_timeline, quality_radar
from visualization.dataset_charts import (
    dataset_health_score,
    dataset_radar,
    missing_primitives_warning,
    primitive_coverage_chart,
    score_distribution,
    summary_table,
)


def _fig_to_html(fig, width: int = 680, height: int = 360) -> str:
    """Render a Plotly figure to a base64 PNG and return an <img> HTML tag.

    This avoids Gradio 6's HTML sanitiser stripping <script> tags (which
    breaks plotly.io.to_html CDN embeds) and the broken gr.Plot renderer.
    """
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
    b64 = base64.b64encode(img_bytes).decode()
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'style="width:100%;border-radius:6px;display:block">'
    )

EXAMPLES_DIR = Path(__file__).parent / "examples"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

PRIMITIVE_EMOJI = {
    PrimitiveType.REACH:     "→ Reach",
    PrimitiveType.GRASP:     "✊ Grasp",
    PrimitiveType.LIFT:      "↑ Lift",
    PrimitiveType.TRANSPORT: "⇒ Transport",
    PrimitiveType.PLACE:     "↓ Place",
    PrimitiveType.RETRACT:   "← Retract",
    PrimitiveType.UNKNOWN:   "? Unknown",
}


def _quality_color(score: float) -> str:
    if score >= 0.65:
        return "#2A9D8F"
    if score >= 0.4:
        return "#E9C46A"
    return "#E76F51"


def _format_primitive_sequence(result: AnalysisResult) -> str:
    if not result.segments:
        return "No primitives detected."
    parts = []
    for ss in result.segments:
        if ss.segment.primitive == PrimitiveType.UNKNOWN:
            continue
        label = PRIMITIVE_EMOJI.get(ss.segment.primitive, ss.segment.primitive.value)
        q = ss.quality.composite
        color = _quality_color(q)
        parts.append(
            f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;margin:2px;'
            f'font-size:0.85rem;background:{color}22;border:1px solid {color}">'
            f"{label} <b>{q:.2f}</b></span>"
        )
    return " ".join(parts) if parts else "<i>Only UNKNOWN segments detected — try a longer clip with clearer hand visibility.</i>"


def _format_vlm(result: AnalysisResult) -> str:
    v = result.vlm_eval
    if v is None or v.skipped:
        note = v.notes if v else "Not evaluated."
        return f"<i style='color:#888'>{note}</i>"
    icon = "✅" if v.task_success else "❌"
    prim_list = ", ".join(v.primitives_observed) if v.primitives_observed else "—"
    return (
        f"<b>{icon} {v.task_description}</b><br>"
        f"Confidence: {v.confidence:.0%} &nbsp;|&nbsp; Primitives: {prim_list}<br>"
        f"<i style='color:#888'>{v.notes}</i>"
    )


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

def analyze_single(video_path: str | None, api_key: str) -> tuple:
    if not video_path:
        raise gr.Error("Please upload a video first.")
    try:
        result = pqa.run(video_path, api_key=api_key or None)
    except Exception as e:
        raise gr.Error(f"Pipeline error: {e}")

    q = result.overall_quality
    color = _quality_color(q.composite)
    score_html = (
        f'<div style="font-size:2rem;font-weight:bold;text-align:center;color:{color}">'
        f"Composite: {q.composite:.0%}</div>"
    )
    # Copy annotated video to Gradio's temp dir so it can serve it
    src = Path(result.annotated_video_path).resolve()
    tmp = Path(tempfile.mktemp(suffix=".mp4"))
    shutil.copy2(src, tmp)
    annotated = str(tmp)
    return (
        annotated,
        score_html,
        _fig_to_html(quality_radar(result)),
        _fig_to_html(primitive_timeline(result.segments, len(result.trajectory))),
        _fig_to_html(per_segment_bars(result.segments)),
        _format_primitive_sequence(result),
        _format_vlm(result),
    )


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

    df = summary_table(results, names)
    export_path = _export_dataset_json(results, names)

    return (
        health_html,
        missing_primitives_warning(results),
        _fig_to_html(dataset_radar(results)),
        _fig_to_html(primitive_coverage_chart(results)),
        _fig_to_html(score_distribution(results, names)),
        df,
        str(export_path),
    )


# ── UI ───────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="PrimitiveQA") as demo:
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
                        video_out = gr.Video(label="Skeleton overlay")
                        score_display = gr.HTML()

                with gr.Row():
                    radar_plot   = gr.HTML(label="Quality radar")
                    timeline_plot = gr.HTML(label="Primitive timeline")

                bars_plot   = gr.HTML(label="Per-segment quality breakdown")
                prim_display = gr.HTML(label="Detected primitive sequence")
                gr.Markdown("### Claude task evaluation")
                vlm_display = gr.HTML()

                run_btn.click(
                    fn=analyze_single,
                    inputs=[video_in, api_key_box],
                    outputs=[video_out, score_display, radar_plot,
                             timeline_plot, bars_plot, prim_display, vlm_display],
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

                    with gr.Column(scale=1):
                        health_display = gr.HTML()
                        missing_display = gr.HTML()

                with gr.Row():
                    dataset_radar_plot   = gr.HTML(label="Average quality radar")
                    coverage_plot        = gr.HTML(label="Primitive coverage")

                dist_plot    = gr.HTML(label="Per-clip composite scores")
                summary_df   = gr.Dataframe(label="Per-clip summary", wrap=True)
                export_file  = gr.File(label="Download dataset_health.json")

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
                        export_file,
                    ],
                )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue()
    app.launch(share=False, show_error=True)
