"""Quality scoring for trajectory segments."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from core.types import QualityScore, Segment, ScoredSegment, Trajectory


def _smoothness(positions: np.ndarray) -> float:
    """1 - normalized jerk. Higher = smoother motion."""
    if len(positions) < 4:
        return 0.5
    # Second derivative of position ≈ acceleration; third ≈ jerk
    vel = np.diff(positions, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=1)
    mean_jerk = float(np.mean(jerk_mag))
    # Normalize: typical jerk in normalized coords ~0.001–0.1
    score = 1.0 / (1.0 + mean_jerk * 50)
    return float(np.clip(score, 0.0, 1.0))


def _path_efficiency(positions: np.ndarray) -> float:
    """Straight-line distance / actual path length. 1 = perfectly straight."""
    if len(positions) < 2:
        return 1.0
    straight = float(np.linalg.norm(positions[-1] - positions[0]))
    actual = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    if actual < 1e-9:
        return 1.0
    return float(np.clip(straight / actual, 0.0, 1.0))


def _decisiveness(velocities: np.ndarray) -> float:
    """Low coefficient of variation in speed = decisive, consistent motion."""
    if len(velocities) < 2:
        return 0.5
    std = float(np.std(velocities))
    mean = float(np.mean(np.abs(velocities))) + 1e-9
    cv = std / mean
    score = 1.0 / (1.0 + cv)
    return float(np.clip(score, 0.0, 1.0))


def score_segment(seg: Segment, trajectory: Trajectory) -> QualityScore:
    pts = trajectory.points[seg.start_idx:seg.end_idx]
    confidences = np.array([p.confidence for p in pts])

    # Only score on detected frames to avoid zero-position artifacts
    detected_mask = confidences > 0
    if detected_mask.sum() >= 2:
        positions = seg.positions[detected_mask]
        velocities = np.array([p.velocity for p in pts])[detected_mask]
    else:
        positions = seg.positions
        velocities = np.array([p.velocity for p in pts])

    s = _smoothness(positions)
    pe = _path_efficiency(positions)
    d = _decisiveness(velocities)
    conf = float(np.mean(confidences)) if len(confidences) else 0.0

    composite = 0.35 * s + 0.25 * pe + 0.25 * d + 0.15 * conf

    return QualityScore(
        smoothness=s,
        path_efficiency=pe,
        decisiveness=d,
        confidence_mean=conf,
        composite=float(np.clip(composite, 0.0, 1.0)),
    )


def score_all(segments: list[Segment], trajectory: Trajectory) -> list[ScoredSegment]:
    return [ScoredSegment(seg, score_segment(seg, trajectory)) for seg in segments]


def overall_quality(scored: list[ScoredSegment], trajectory: Trajectory) -> QualityScore:
    """Weighted average across all segments, weighted by segment length."""
    if not scored:
        pts = trajectory.points
        conf = float(np.mean([p.confidence for p in pts])) if pts else 0.0
        return QualityScore(0.5, 0.5, 0.5, conf, 0.5)

    weights = np.array([s.segment.duration_frames() for s in scored], dtype=float)
    weights /= weights.sum() + 1e-9

    def wavg(attr: str) -> float:
        return float(np.sum([getattr(s.quality, attr) * w for s, w in zip(scored, weights)]))

    s = wavg("smoothness")
    pe = wavg("path_efficiency")
    d = wavg("decisiveness")
    c = wavg("confidence_mean")
    comp = 0.35 * s + 0.25 * pe + 0.25 * d + 0.15 * c

    return QualityScore(
        smoothness=s,
        path_efficiency=pe,
        decisiveness=d,
        confidence_mean=c,
        composite=float(np.clip(comp, 0.0, 1.0)),
    )
