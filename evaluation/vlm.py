"""Claude video-reasoning evaluation of manipulation primitives."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np

from core.types import VLMEvaluation, VLMSegment

N_SAMPLE_FRAMES = 20

SYSTEM_PROMPT = """You are a robotics training-data quality evaluator.
You analyse sequences of frames from hand manipulation videos and return
structured JSON — nothing else."""

USER_PROMPT = """These {n} frames are evenly sampled from a {duration:.1f}s manipulation video.
Each frame has its timestamp burned into the top-left corner (yellow text).

Identify every manipulation primitive that occurs, in order, with the timestamps
at which each one starts and ends. Then rate its execution quality.

Return ONLY a JSON object matching this schema — no markdown, no extra keys:

{{
  "task_description": "one sentence describing the overall manipulation task",
  "task_success": true or false,
  "confidence": float 0–1,
  "segments": [
    {{
      "primitive": "reach | grasp | lift | transport | place | retract",
      "start_sec": float,
      "end_sec": float,
      "quality": "good | ok | poor",
      "note": "one sentence — what happened and any specific quality issue"
    }}
  ],
  "overall_notes": "one sentence about training-data quality (gaps, blur, occlusion, etc.)"
}}

Primitive definitions:
  reach     — hand/arm extends toward the target object
  grasp     — fingers close around the object
  lift      — object is raised off the surface
  transport — object carried laterally to the target location
  place     — object set down at the target
  retract   — hand withdraws after release

Quality rubric:
  good — smooth, direct, deliberate execution; no corrections
  ok   — minor hesitation, slight path deviation, or partial occlusion
  poor — jerky, wandering, highly uncertain, or major occlusion"""


def _sample_frames(
    video_path: str | Path, n: int = N_SAMPLE_FRAMES
) -> tuple[list[bytes], float]:
    """Sample n evenly-spaced frames, burn timestamp, return (jpeg_bytes, duration_sec)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total / fps

    if total < 1:
        cap.release()
        return [], 0.0

    indices = np.linspace(0, total - 1, min(n, total), dtype=int)
    frames: list[bytes] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        t = idx / fps
        # Yellow timestamp so Claude can read temporal position of each frame
        cv2.putText(
            frame, f"t={t:.1f}s",
            (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2,
            cv2.LINE_AA,
        )
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frames.append(buf.tobytes())

    cap.release()
    return frames, duration


def _build_content(video_path: str | Path) -> tuple[list[dict], float]:
    """Shared frame prep: returns (content_blocks, duration_sec)."""
    frame_bytes, duration = _sample_frames(video_path)
    content: list[dict] = []
    for fb in frame_bytes:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(fb).decode(),
            },
        })
    content.append({
        "type": "text",
        "text": USER_PROMPT.format(n=len(frame_bytes), duration=duration),
    })
    return content, duration


def stream_raw(
    video_path: str | Path, api_key: str | None = None
):
    """Generator that yields Claude's response text one chunk at a time.

    Yields str chunks while streaming; when the stream is complete the
    generator returns (StopIteration) — the caller can then call
    parse_response() on the accumulated text to get a VLMEvaluation.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return

    try:
        import anthropic
    except ImportError:
        return

    content, _ = _build_content(video_path)
    if not content:
        return

    try:
        client = anthropic.Anthropic(api_key=key, max_retries=1)
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception:  # noqa: BLE001
        return


def parse_response(raw: str) -> VLMEvaluation:
    """Parse accumulated Claude output into a VLMEvaluation."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _skipped(f"Could not parse Claude response: {raw[:120]}")

    segments: list[VLMSegment] = []
    for s in data.get("segments", []):
        try:
            segments.append(VLMSegment(
                primitive=str(s.get("primitive", "unknown")).lower(),
                start_sec=float(s.get("start_sec", 0)),
                end_sec=float(s.get("end_sec", 0)),
                quality=str(s.get("quality", "ok")).lower(),
                note=str(s.get("note", "")),
            ))
        except (TypeError, ValueError):
            continue

    return VLMEvaluation(
        task_description=data.get("task_description", ""),
        task_success=bool(data.get("task_success", False)),
        confidence=float(data.get("confidence", 0.0)),
        primitives_observed=[s.primitive for s in segments],
        notes=data.get("overall_notes", ""),
        segments=segments,
        skipped=False,
    )


def _skipped(note: str) -> VLMEvaluation:
    return VLMEvaluation(
        task_description="",
        task_success=False,
        confidence=0.0,
        primitives_observed=[],
        notes=note,
        skipped=True,
    )


def evaluate(video_path: str | Path, api_key: str | None = None) -> VLMEvaluation:
    """Non-streaming convenience wrapper — collects the full stream then parses."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return _skipped("Set ANTHROPIC_API_KEY to enable Claude video reasoning.")

    accumulated = "".join(stream_raw(video_path, api_key=key))
    if not accumulated:
        return _skipped("Claude returned no output.")
    return parse_response(accumulated)
