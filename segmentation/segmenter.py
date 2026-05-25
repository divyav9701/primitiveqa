"""Segment a Trajectory into manipulation primitives using heuristics."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from core.types import PrimitiveType, Segment, Trajectory

# Thresholds
GRIPPER_CLOSE_THRESH = 0.55   # gripper_state > this → hand is closing/closed
GRIPPER_OPEN_THRESH = 0.35    # gripper_state < this → hand is opening/open
MIN_SEGMENT_FRAMES = 3        # discard segments shorter than this


def _smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    if len(arr) < window:
        return arr.copy()
    return savgol_filter(arr, window_length=min(window, len(arr) | 1), polyorder=2)


def _classify_frame(
    pos: np.ndarray,       # (3,) current wrist position (x, y, z)
    prev_pos: np.ndarray,  # (3,)
    gripper: float,
    prev_gripper: float,
    vel: float,
    z_up_bias: float,      # positive = moving up (lower y in image coords, or higher z)
) -> PrimitiveType:
    """
    Heuristic single-frame primitive label.

    MediaPipe wrist coords: x ∈ [0,1] left→right, y ∈ [0,1] top→bottom, z depth.
    'Lift' = wrist y decreasing (moving toward top of frame).
    'Place' = wrist y increasing (moving toward bottom of frame).
    """
    dy = pos[1] - prev_pos[1]   # positive = moving down in image
    dx_mag = abs(pos[0] - prev_pos[0])
    moving = vel > 0.02          # normalized velocity threshold

    if gripper < GRIPPER_OPEN_THRESH and prev_gripper < GRIPPER_OPEN_THRESH:
        if moving:
            return PrimitiveType.REACH
        return PrimitiveType.UNKNOWN

    if gripper > GRIPPER_CLOSE_THRESH and prev_gripper < GRIPPER_CLOSE_THRESH:
        return PrimitiveType.GRASP

    if gripper > GRIPPER_CLOSE_THRESH:
        if dy < -0.01 and moving:       # moving up
            return PrimitiveType.LIFT
        if dy > 0.01 and moving:        # moving down
            return PrimitiveType.PLACE
        if dx_mag > 0.01 and moving:
            return PrimitiveType.TRANSPORT
        return PrimitiveType.TRANSPORT  # stationary grasp → still transport

    if gripper < GRIPPER_OPEN_THRESH and prev_gripper > GRIPPER_CLOSE_THRESH:
        return PrimitiveType.RETRACT

    return PrimitiveType.UNKNOWN


def _merge_segments(labels: list[PrimitiveType], min_len: int) -> list[tuple[int, int, PrimitiveType]]:
    """Merge consecutive same-label frames into (start, end, label) spans, drop short ones."""
    if not labels:
        return []

    spans: list[tuple[int, int, PrimitiveType]] = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            spans.append((start, i, labels[start]))
            start = i
    spans.append((start, len(labels), labels[start]))

    # Drop segments shorter than min_len by merging into neighbors
    merged = True
    while merged:
        merged = False
        new_spans = []
        i = 0
        while i < len(spans):
            s, e, lbl = spans[i]
            if (e - s) < min_len and len(spans) > 1:
                # absorb into previous if exists, else next
                if new_spans:
                    ps, pe, pl = new_spans[-1]
                    new_spans[-1] = (ps, e, pl)
                elif i + 1 < len(spans):
                    ns, ne, nl = spans[i + 1]
                    spans[i + 1] = (s, ne, nl)
                    i += 1
                    continue
                merged = True
            else:
                new_spans.append((s, e, lbl))
            i += 1
        spans = new_spans

    return spans


def segment(trajectory: Trajectory) -> list[Segment]:
    """Decompose trajectory into a list of primitive Segments."""
    pts = trajectory.points
    n = len(pts)
    if n < 2:
        return []

    positions = trajectory.positions()
    gripper = _smooth(trajectory.gripper_states())
    velocities = _smooth(trajectory.velocities())

    labels: list[PrimitiveType] = [PrimitiveType.UNKNOWN]
    for i in range(1, n):
        lbl = _classify_frame(
            pos=positions[i],
            prev_pos=positions[i - 1],
            gripper=float(gripper[i]),
            prev_gripper=float(gripper[i - 1]),
            vel=float(velocities[i]),
            z_up_bias=float(positions[i - 1][1] - positions[i][1]),
        )
        labels.append(lbl)

    spans = _merge_segments(labels, MIN_SEGMENT_FRAMES)

    segments: list[Segment] = []
    for start, end, prim in spans:
        segments.append(Segment(
            primitive=prim,
            start_idx=start,
            end_idx=end,
            positions=positions[start:end],
            gripper=gripper[start:end],
        ))

    return segments
