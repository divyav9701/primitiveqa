from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class PrimitiveType(str, Enum):
    REACH = "reach"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    PLACE = "place"
    RETRACT = "retract"
    UNKNOWN = "unknown"


@dataclass
class TrajectoryPoint:
    timestamp: float
    x: float
    y: float
    z: float
    gripper_state: float  # 0=open, 1=closed
    velocity: float
    confidence: float


@dataclass
class Trajectory:
    points: list[TrajectoryPoint]
    source: str = "phone_video"
    fps: float = 30.0

    def __len__(self) -> int:
        return len(self.points)

    def positions(self) -> np.ndarray:
        return np.array([[p.x, p.y, p.z] for p in self.points])

    def gripper_states(self) -> np.ndarray:
        return np.array([p.gripper_state for p in self.points])

    def velocities(self) -> np.ndarray:
        return np.array([p.velocity for p in self.points])

    def confidences(self) -> np.ndarray:
        return np.array([p.confidence for p in self.points])


@dataclass
class Segment:
    primitive: PrimitiveType
    start_idx: int
    end_idx: int
    positions: np.ndarray   # (N, 3)
    gripper: np.ndarray     # (N,)

    def duration_frames(self) -> int:
        return self.end_idx - self.start_idx


@dataclass
class QualityScore:
    smoothness: float       # 0-1, trajectory smoothness
    path_efficiency: float  # 0-1, straight-line vs actual path
    decisiveness: float     # 0-1, low velocity variance = decisive
    confidence_mean: float  # 0-1, mean MediaPipe detection confidence
    composite: float        # weighted combination

    def to_dict(self) -> dict:
        return {
            "smoothness": round(self.smoothness, 3),
            "path_efficiency": round(self.path_efficiency, 3),
            "decisiveness": round(self.decisiveness, 3),
            "confidence": round(self.confidence_mean, 3),
            "composite": round(self.composite, 3),
        }


@dataclass
class ScoredSegment:
    segment: Segment
    quality: QualityScore


@dataclass
class VLMSegment:
    """One primitive identified by the VLM, with timestamps and a quality note."""
    primitive: str      # "reach", "grasp", "lift", "transport", "place", "retract"
    start_sec: float
    end_sec: float
    quality: str        # "good", "ok", "poor"
    note: str           # one-sentence reasoning from the model


@dataclass
class VLMEvaluation:
    task_description: str
    task_success: bool
    confidence: float
    primitives_observed: list[str]
    notes: str
    skipped: bool = False
    segments: list[VLMSegment] = field(default_factory=list)


@dataclass
class AnalysisResult:
    trajectory: Trajectory
    segments: list[ScoredSegment]
    overall_quality: QualityScore
    vlm_eval: Optional[VLMEvaluation]
    annotated_video_path: Optional[str]
    export_path: Optional[str] = None
