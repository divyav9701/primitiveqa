"""PrimitiveQA — Gradio web app."""

from __future__ import annotations

import os
import json
import shutil
import tempfile
from pathlib import Path

import gradio as gr

import pipeline as pqa
from core.types import AnalysisResult, PrimitiveType
from visualization.charts import per_segment_bars, primitive_timeline, quality_radar

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

CSS = """
.composite-score { font-size: 2rem; font-weight: bold; text-align: center; }
.primitive-tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
                 margin: 2px; font-size: 0.85rem; }
"""


def _quality_color(score: float) -> str:
    if score >= 0.75:
        return "#2A9D8F"
    if score >= 0.5:
        return "#E9C46A"
    return "#E76F51"


def _format_primitive_sequence(result: AnalysisResult) -> str:
    if not result.segments:
        return "No primitives detected."
    parts = []
    for ss in result.segments:
        label = PRIMITIVE_EMOJI.get(ss.segment.primitive, ss.segment.primitive.value)
        q = ss.quality.composite
        color = _quality_color(q)
        parts.append(
            f'<span class="primitive-tag" style="background:{color}22;border:1px solid {color}">'
            f"{label} <b>{q:.2f}</b></span>"
        )
    return " ".join(parts)


def _format_vlm(result: AnalysisResult) -> str:
    v = result.vlm_eval
    if v is None or v.skipped:
        note = v.notes if v else "Not evaluated."
        return f"<i style='color:#888'>{note}</i>"
    icon = "✅" if v.task_success else "❌"
    prim_list = ", ".join(v.primitives_observed) if v.primitives_observed else "—"
    return (
        f"<b>{icon} {v.task_description}</b><br>"
        f"Confidence: {v.confidence:.0%} &nbsp;|&nbsp; "
        f"Primitives: {prim_list}<br>"
        f"<i style='color:#888'>{v.notes}</i>"
    )


def _export_json(result: AnalysisResult) -> Path:
    data = {
        "overall_quality": result.overall_quality.to_dict(),
        "segments": [
            {
                "primitive": ss.segment.primitive.value,
                "start_frame": ss.segment.start_idx,
                "end_frame": ss.segment.end_idx,
                "duration_frames": ss.segment.duration_frames(),
                "quality": ss.quality.to_dict(),
            }
            for ss in result.segments
        ],
        "vlm_evaluation": {
            "task_description": result.vlm_eval.task_description if result.vlm_eval else "",
            "task_success": result.vlm_eval.task_success if result.vlm_eval else None,
            "confidence": result.vlm_eval.confidence if result.vlm_eval else None,
            "primitives_observed": result.vlm_eval.primitives_observed if result.vlm_eval else [],
        } if result.vlm_eval else None,
        "trajectory_length": len(result.trajectory),
        "source": result.trajectory.source,
    }
    out = OUTPUT_DIR / "primitiveqa_result.json"
    out.write_text(json.dumps(data, indent=2))
    return out


def process(video_path: str | None, api_key: str) -> tuple:
    if not video_path:
        raise gr.Error("Please upload a video first.")

    try:
        result = pqa.run(video_path, api_key=api_key or None)
    except Exception as e:
        raise gr.Error(f"Pipeline error: {e}")

    radar = quality_radar(result)
    timeline = primitive_timeline(result.segments, len(result.trajectory))
    bars = per_segment_bars(result.segments)

    composite = result.overall_quality.composite
    color = _quality_color(composite)
    score_html = (
        f'<div class="composite-score" style="color:{color}">'
        f"Composite Quality: {composite:.0%}</div>"
    )

    prim_html = _format_primitive_sequence(result)
    vlm_html = _format_vlm(result)

    export_path = _export_json(result)

    annotated = result.annotated_video_path

    return (
        annotated,
        score_html,
        radar,
        timeline,
        bars,
        prim_html,
        vlm_html,
        str(export_path),
    )


def build_ui() -> gr.Blocks:
    example_files = sorted(EXAMPLES_DIR.glob("*.mp4")) + sorted(EXAMPLES_DIR.glob("*.mov"))

    with gr.Blocks(css=CSS, title="PrimitiveQA") as demo:
        gr.Markdown("# PrimitiveQA\n**The quality layer for physical AI data.** "
                    "Upload a hand manipulation video → get primitive decomposition + quality scores.")

        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(label="Upload manipulation video", sources=["upload"])

                if example_files:
                    gr.Examples(
                        examples=[[str(p)] for p in example_files[:3]],
                        inputs=[video_in],
                        label="Example clips",
                    )

                api_key_box = gr.Textbox(
                    label="Anthropic API key (optional — enables Claude evaluation)",
                    placeholder="sk-ant-...",
                    type="password",
                    value=os.environ.get("ANTHROPIC_API_KEY", ""),
                )
                run_btn = gr.Button("Analyze", variant="primary", size="lg")

            with gr.Column(scale=1):
                video_out = gr.Video(label="Annotated skeleton overlay")
                score_display = gr.HTML(label="Overall quality")

        with gr.Row():
            radar_plot = gr.Plot(label="Quality radar")
            timeline_plot = gr.Plot(label="Primitive timeline")

        bars_plot = gr.Plot(label="Per-segment quality breakdown")

        gr.Markdown("### Detected primitive sequence")
        prim_display = gr.HTML()

        gr.Markdown("### Claude task evaluation")
        vlm_display = gr.HTML()

        export_file = gr.File(label="Download JSON export")

        run_btn.click(
            fn=process,
            inputs=[video_in, api_key_box],
            outputs=[
                video_out,
                score_display,
                radar_plot,
                timeline_plot,
                bars_plot,
                prim_display,
                vlm_display,
                export_file,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(share=False)
