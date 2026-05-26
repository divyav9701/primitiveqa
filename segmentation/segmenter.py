"""Segment a Trajectory into manipulation primitives using heuristics."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from core.types import PrimitiveType, Segment, Trajectory

# Thresholds — adaptive, computed per-trajectory
MIN_SEGMENT_FRAMES = 3        # discard segments shorter than this


def _smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    if len(arr) < window:
        return arr.copy()
    return savgol_filter(arr, window_length=min(window, len(arr) | 1), polyorder=2)


def _classify_frame(
    pos: np.ndarray,
    prev_pos: np.ndarray,
    gripper: float,
    prev_gripper: float,
    vel: float,
    z_up_bias: float,
    gripper_close: float,
    gripper_open: float,
) -> PrimitiveType:
    """
    Heuristic single-frame primitive label with adaptive thresholds.

    MediaPipe wrist coords: x ∈ [0,1] left→right, y ∈ [0,1] top→bottom.
    'Lift' = wrist y decreasing (moving toward top of frame).
    'Place' = wrist y increasing (moving toward bottom of frame).
    """
    dy = pos[1] - prev_pos[1]
    dx_mag = abs(pos[0] - prev_pos[0])
    moving = vel > 0.02

    if gripper < gripper_open and prev_gripper < gripper_open:
        if moving:
            return PrimitiveType.REACH
        return PrimitiveType.UNKNOWN

    if gripper > gripper_close and prev_gripper < gripper_close:
        return PrimitiveType.GRASP

    if gripper > gripper_close:
        if dy < -0.01 and moving:
            return PrimitiveType.LIFT
        if dy > 0.01 and moving:
            return PrimitiveType.PLACE
        if dx_mag > 0.01 and moving:
            return PrimitiveType.TRANSPORT
        return PrimitiveType.TRANSPORT

    if gripper < gripper_open and prev_gripper > gripper_close:
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


def _adaptive_thresholds(gripper: np.ndarray) -> tuple[float, float]:
    """Compute close/open thresholds relative to observed range."""
    g_min = float(np.percentile(gripper, 10))
    g_max = float(np.percentile(gripper, 90))
    span = max(g_max - g_min, 0.05)
    close = g_min + span * 0.65
    open_ = g_min + span * 0.35
    return close, open_


def segment(trajectory: Trajectory) -> list[Segment]:
    """Decompose trajectory into a list of primitive Segments."""
    pts = trajectory.points
    n = len(pts)
    if n < 2:
        return []

    positions = trajectory.positions()
    gripper_raw = trajectory.gripper_states()
    confidences = trajectory.confidences()
    gripper = _smooth(gripper_raw)
    velocities = _smooth(trajectory.velocities())

    # Interpolate positions for zero-confidence frames so metrics aren't trashed
    for i in range(n):
        if confidences[i] == 0:
            # find nearest detected neighbors
            prev_i = next((j for j in range(i - 1, -1, -1) if confidences[j] > 0), None)
            next_i = next((j for j in range(i + 1, n) if confidences[j] > 0), None)
            if prev_i is not None and next_i is not None:
                alpha = (i - prev_i) / (next_i - prev_i)
                positions[i] = (1 - alpha) * positions[prev_i] + alpha * positions[next_i]
            elif prev_i is not None:
                positions[i] = positions[prev_i]
            elif next_i is not None:
                positions[i] = positions[next_i]

    gripper_close, gripper_open = _adaptive_thresholds(gripper)

    labels: list[PrimitiveType] = [PrimitiveType.UNKNOWN]
    for i in range(1, n):
        lbl = _classify_frame(
            pos=positions[i],
            prev_pos=positions[i - 1],
            gripper=float(gripper[i]),
            prev_gripper=float(gripper[i - 1]),
            vel=float(velocities[i]),
            z_up_bias=float(positions[i - 1][1] - positions[i][1]),
            gripper_close=gripper_close,
            gripper_open=gripper_open,
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
