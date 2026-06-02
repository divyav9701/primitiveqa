"""MediaPipe Hand Landmarker source — extracts a Trajectory from a phone video."""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components import containers as mp_containers

from core.types import Trajectory, TrajectoryPoint

MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"


def _reencode_h264(src: Path) -> Path:
    """Re-encode an OpenCV (mp4v) clip to browser-playable H.264, in place.

    HTML5 <video> (and Gradio's player) can't decode mp4v, so the annotated
    overlay shows a NaN duration and won't play. If ffmpeg is unavailable we
    keep the original file rather than fail the pipeline.
    """
    ffmpeg = shutil.which("ffmpeg") or "/opt/anaconda3/bin/ffmpeg"
    if not Path(ffmpeg).exists():
        return src
    dst = src.parent / f"{src.stem}_h264.mp4"
    # Try encoders in order — builds vary (libx264 is often absent; macOS ships
    # the hardware videotoolbox encoder; libopenh264 is a common fallback).
    for encoder in ("libx264", "h264_videotoolbox", "libopenh264"):
        try:
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
                 "-c:v", encoder, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 "-an", str(dst)],
                check=True,
            )
        except (subprocess.CalledProcessError, OSError):
            continue
        if dst.exists() and dst.stat().st_size > 0:
            return dst
    return src

# Landmark indices (MediaPipe 21-point hand model)
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17


def _finger_curl(landmarks) -> float:
    """0 = fully open, 1 = fully closed."""
    pairs = [
        (INDEX_TIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_MCP),
        (RING_TIP, RING_MCP),
        (PINKY_TIP, PINKY_MCP),
    ]
    wrist = landmarks[WRIST]
    curls = []
    for tip_i, mcp_i in pairs:
        tip = landmarks[tip_i]
        mcp = landmarks[mcp_i]
        tip_mcp = math.dist([tip.x, tip.y, tip.z], [mcp.x, mcp.y, mcp.z])
        mcp_wrist = math.dist([mcp.x, mcp.y, mcp.z], [wrist.x, wrist.y, wrist.z])
        ref = max(mcp_wrist, 1e-6)
        curl = 1.0 - min(tip_mcp / ref, 1.0)
        curls.append(curl)
    return float(np.mean(curls))


def _draw_connections(frame: np.ndarray, landmarks, connections) -> None:
    """Draw hand skeleton overlay on frame (in-place)."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for conn in connections:
        a, b = conn.start, conn.end
        cv2.line(frame, pts[a], pts[b], (0, 220, 100), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        cv2.circle(frame, pt, 4, (0, 180, 80), 1)


def process_video(
    video_path: str | Path,
    target_fps: float = 15.0,
) -> tuple[Trajectory, Path]:
    """
    Run MediaPipe Hand Landmarker on every sampled frame.
    Returns (Trajectory, annotated_video_path).
    """
    video_path = Path(video_path)
    out_path = video_path.parent / f"{video_path.stem}_annotated.mp4"

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Hand landmarker model not found at {MODEL_PATH}. "
            "Run: curl -L -o models/hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(src_fps / max(target_fps, 1.0))))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, target_fps, (width, height))

    points: list[TrajectoryPoint] = []
    prev_pos: np.ndarray | None = None
    frame_idx = 0
    t = 0.0
    dt = 1.0 / target_fps

    connections = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                wrist = lm[WRIST]
                pos = np.array([wrist.x, wrist.y, wrist.z])

                vel = 0.0
                if prev_pos is not None:
                    vel = float(np.linalg.norm(pos - prev_pos) / dt)
                prev_pos = pos

                conf = float(result.handedness[0][0].score) if result.handedness else 0.9
                gripper = _finger_curl(lm)

                points.append(TrajectoryPoint(
                    timestamp=t,
                    x=float(wrist.x),
                    y=float(wrist.y),
                    z=float(wrist.z),
                    gripper_state=gripper,
                    velocity=vel,
                    confidence=conf,
                ))
                _draw_connections(frame, lm, connections)
            else:
                prev_pos = None
                points.append(TrajectoryPoint(
                    timestamp=t, x=0.0, y=0.0, z=0.0,
                    gripper_state=0.0, velocity=0.0, confidence=0.0,
                ))

            writer.write(frame)
            t += dt
            frame_idx += 1

    cap.release()
    writer.release()

    out_path = _reencode_h264(out_path)

    return Trajectory(points=points, source="phone_video", fps=target_fps), out_path
