"""Orchestrator: runs the full PrimitiveQA pipeline on a video file."""

from __future__ import annotations

from pathlib import Path

from core.types import AnalysisResult
from evaluation.vlm import evaluate
from scoring.scorer import overall_quality, score_all
from segmentation.segmenter import segment
from sources.phone_video import process_video


def run(
    video_path: str | Path,
    api_key: str | None = None,
    target_fps: float = 15.0,
    skip_vlm: bool = False,
) -> AnalysisResult:
    video_path = Path(video_path)

    trajectory, annotated_path = process_video(video_path, target_fps=target_fps)

    segments = segment(trajectory)

    scored = score_all(segments, trajectory)

    overall = overall_quality(scored, trajectory)

    vlm_eval = None if skip_vlm else evaluate(video_path, api_key=api_key)

    return AnalysisResult(
        trajectory=trajectory,
        segments=scored,
        overall_quality=overall,
        vlm_eval=vlm_eval,
        annotated_video_path=str(annotated_path),
    )
