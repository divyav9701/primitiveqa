"""Claude vision evaluation of manipulation task success."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np

from core.types import VLMEvaluation

N_SAMPLE_FRAMES = 6

SYSTEM_PROMPT = """You are a robotics data quality evaluator. You analyze short video clips of
hand manipulation tasks and return structured JSON assessments."""

USER_PROMPT = """These {n} frames are evenly sampled from a manipulation video.
Analyze the hand motion and return ONLY a JSON object with this exact schema:

{{
  "task_description": "one sentence describing the manipulation task",
  "task_success": true or false,
  "confidence": float between 0 and 1,
  "primitives_observed": ["list", "of", "observed", "primitives"],
  "notes": "one sentence about data quality issues if any"
}}

Primitives to look for: reach, grasp, lift, transport, place, retract.
Do not include any text outside the JSON."""


def _sample_frames(video_path: str | Path, n: int = N_SAMPLE_FRAMES) -> list[bytes]:
    """Sample n evenly-spaced frames from video, return as JPEG bytes."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append(buf.tobytes())

    cap.release()
    return frames


def evaluate(video_path: str | Path, api_key: str | None = None) -> VLMEvaluation:
    """Call Claude vision to evaluate task success. Returns placeholder if no API key."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return VLMEvaluation(
            task_description="(API key not set — skipped)",
            task_success=False,
            confidence=0.0,
            primitives_observed=[],
            notes="Set ANTHROPIC_API_KEY to enable Claude evaluation.",
            skipped=True,
        )

    try:
        import anthropic
    except ImportError:
        return VLMEvaluation(
            task_description="(anthropic package not installed)",
            task_success=False,
            confidence=0.0,
            primitives_observed=[],
            notes="pip install anthropic",
            skipped=True,
        )

    frame_bytes = _sample_frames(video_path, N_SAMPLE_FRAMES)
    if not frame_bytes:
        return VLMEvaluation(
            task_description="(no frames extracted)",
            task_success=False,
            confidence=0.0,
            primitives_observed=[],
            notes="Video could not be read.",
            skipped=True,
        )

    content: list[dict] = []
    for fb in frame_bytes:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(fb).decode("utf-8"),
            },
        })
    content.append({
        "type": "text",
        "text": USER_PROMPT.format(n=len(frame_bytes)),
    })

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)
    return VLMEvaluation(
        task_description=data.get("task_description", ""),
        task_success=bool(data.get("task_success", False)),
        confidence=float(data.get("confidence", 0.0)),
        primitives_observed=data.get("primitives_observed", []),
        notes=data.get("notes", ""),
        skipped=False,
    )
