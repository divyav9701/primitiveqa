"""Segment a Trajectory into manipulation primitives using motion direction."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from core.types import PrimitiveType, Segment, Trajectory

MIN_SEGMENT_FRAMES = 3


def _smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    n = len(arr)
    if n < 4:
        return arr.copy()
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return arr.copy()
    return savgol_filter(arr, window_length=w, polyorder=2)


def _interpolate_gaps(positions: np.ndarray, confidences: np.ndarray) -> np.ndarray:
    """Fill zero-confidence frames by linear interpolation from neighbours."""
    pos = positions.copy()
    n = len(pos)
    for i in range(n):
        if confidences[i] > 0:
            continue
        prev_i = next((j for j in range(i - 1, -1, -1) if confidences[j] > 0), None)
        next_i = next((j for j in range(i + 1, n)     if confidences[j] > 0), None)
        if prev_i is not None and next_i is not None:
            alpha = (i - prev_i) / (next_i - prev_i)
            pos[i] = (1 - alpha) * pos[prev_i] + alpha * pos[next_i]
        elif prev_i is not None:
            pos[i] = pos[prev_i]
        elif next_i is not None:
            pos[i] = pos[next_i]
    return pos


def _classify_frame(
    dy: float,       # change in wrist y (positive = moving DOWN in image)
    dx: float,       # change in wrist x
    speed: float,    # magnitude of wrist movement per frame
    seq_pos: float,  # position in trajectory 0=start 1=end
) -> PrimitiveType:
    """
    Motion-direction classifier — does NOT use gripper state.

    MediaPipe y: 0=top, 1=bottom of frame.
    Lift  = wrist moves UP   → dy < 0
    Place = wrist moves DOWN → dy > 0
    """

    SPEED_STILL   = 0.005   # nearly stationary
    SPEED_MOVING  = 0.012   # clearly moving
    VERT_BIAS     = 1.4     # vertical must dominate by this factor to call lift/place

    if speed < SPEED_STILL:
        # Stationary — grasp if mid-sequence, otherwise unknown
        if 0.15 < seq_pos < 0.85:
            return PrimitiveType.GRASP
        return PrimitiveType.UNKNOWN

    # Determine dominant direction
    vert_dom = abs(dy) > abs(dx) * (1 / VERT_BIAS)

    if vert_dom and speed > SPEED_MOVING:
        if dy < -0.008:
            return PrimitiveType.LIFT
        if dy > 0.008:
            return PrimitiveType.PLACE

    if speed > SPEED_MOVING:
        if seq_pos < 0.25:
            return PrimitiveType.REACH
        if seq_pos > 0.75:
            return PrimitiveType.RETRACT
        return PrimitiveType.TRANSPORT

    return PrimitiveType.UNKNOWN


def _merge_segments(
    labels: list[PrimitiveType], min_len: int
) -> list[tuple[int, int, PrimitiveType]]:
    if not labels:
        return []

    spans: list[tuple[int, int, PrimitiveType]] = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            spans.append((start, i, labels[start]))
            start = i
    spans.append((start, len(labels), labels[start]))

    # Absorb short spans into their longer neighbour
    changed = True
    while changed:
        changed = False
        new_spans: list[tuple[int, int, PrimitiveType]] = []
        i = 0
        while i < len(spans):
            s, e, lbl = spans[i]
            if (e - s) < min_len and len(spans) > 1:
                if new_spans:
                    ps, pe, pl = new_spans[-1]
                    new_spans[-1] = (ps, e, pl)
                elif i + 1 < len(spans):
                    ns, ne, nl = spans[i + 1]
                    spans[i + 1] = (s, ne, nl)
                    i += 1
                    continue
                changed = True
            else:
                new_spans.append((s, e, lbl))
            i += 1
        spans = new_spans

    return spans


def segment(trajectory: Trajectory) -> list[Segment]:
    """Decompose trajectory into primitive Segments."""
    pts = trajectory.points
    n = len(pts)
    if n < 2:
        return []

    confidences = trajectory.confidences()
    raw_positions = trajectory.positions()
    positions = _interpolate_gaps(raw_positions, confidences)

    # Smooth wrist x and y independently
    xs = _smooth(positions[:, 0])
    ys = _smooth(positions[:, 1])

    labels: list[PrimitiveType] = [PrimitiveType.UNKNOWN]

    for i in range(1, n):
        dy    = float(ys[i] - ys[i - 1])
        dx    = float(xs[i] - xs[i - 1])
        speed = float(np.sqrt(dx ** 2 + dy ** 2))
        seq_pos = i / (n - 1)

        labels.append(_classify_frame(dy, dx, speed, seq_pos))

    spans = _merge_segments(labels, MIN_SEGMENT_FRAMES)

    segments: list[Segment] = []
    for start, end, prim in spans:
        segments.append(Segment(
            primitive=prim,
            start_idx=start,
            end_idx=end,
            positions=positions[start:end],
            gripper=trajectory.gripper_states()[start:end],
        ))

    return segments
