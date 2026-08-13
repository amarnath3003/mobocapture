from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import mediapipe as mp
import pyarrow as pa
import pyarrow.parquet as pq

from mobocapture.assets import ensure_hand_landmarker_model
from mobocapture.models import EpistemicStatus, ProvenanceClass
from mobocapture.processors.base import Processor, ProcessorResult
from mobocapture.schemas import HAND_FRAMES_SCHEMA, HANDS_SCHEMA
from mobocapture.session import SessionWorkspace


@dataclass
class _Track:
    track_id: str
    wrist_x: float
    wrist_y: float
    last_frame: int
    side_votes: dict[str, float] = field(default_factory=dict)

    @property
    def side(self) -> str:
        if not self.side_votes:
            return "unknown"
        return max(self.side_votes, key=self.side_votes.get)


class _TrackAssigner:
    def __init__(self, maximum_age_frames: int = 8, maximum_wrist_distance: float = 0.25):
        self.maximum_age_frames = maximum_age_frames
        self.maximum_wrist_distance = maximum_wrist_distance
        self._tracks: list[_Track] = []
        self._next_id = 1

    def assign(
        self,
        frame_index: int,
        detections: list[tuple[float, float, str, float]],
    ) -> list[tuple[str, str]]:
        self._tracks = [
            track
            for track in self._tracks
            if frame_index - track.last_frame <= self.maximum_age_frames
        ]
        assignments: list[tuple[str, str]] = []
        used_tracks: set[str] = set()
        for wrist_x, wrist_y, detected_side, side_confidence in detections:
            candidates: list[tuple[float, _Track]] = []
            for track in self._tracks:
                if track.track_id in used_tracks:
                    continue
                distance = math.hypot(wrist_x - track.wrist_x, wrist_y - track.wrist_y)
                if distance > self.maximum_wrist_distance:
                    continue
                side_penalty = (
                    0.4
                    if detected_side != "unknown"
                    and track.side != "unknown"
                    and detected_side != track.side
                    else 0.0
                )
                candidates.append((distance + side_penalty, track))
            if candidates:
                _, track = min(candidates, key=lambda item: item[0])
            else:
                track = _Track(
                    track_id=f"hand-{self._next_id:04d}",
                    wrist_x=wrist_x,
                    wrist_y=wrist_y,
                    last_frame=frame_index,
                )
                self._next_id += 1
                self._tracks.append(track)
            track.wrist_x = wrist_x
            track.wrist_y = wrist_y
            track.last_frame = frame_index
            track.side_votes[detected_side] = track.side_votes.get(detected_side, 0.0) + side_confidence
            used_tracks.add(track.track_id)
            assignments.append((track.track_id, track.side))
        return assignments


def _category(result: Any, index: int) -> tuple[str, float]:
    categories = result.handedness[index] if index < len(result.handedness) else []
    if not categories:
        return "unknown", 0.0
    category = categories[0]
    side = (category.category_name or "unknown").lower()
    return side if side in {"left", "right"} else "unknown", float(category.score or 0.0)


class HandTrackerProcessor(Processor):
    processor_id = "hand_tracker"

    def process(self, workspace: SessionWorkspace) -> ProcessorResult:
        model_path, model_hash, model_source = ensure_hand_landmarker_model()
        frame_table = pq.read_table(workspace.derived / "frame_index.parquet")
        timestamps = frame_table.column("timestamp_ns").to_pylist()
        capture = cv2.VideoCapture(str(workspace.raw_video))
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the copied input video for hand tracking")

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        assigner = _TrackAssigner()
        hand_rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        previous_inference_ms = -1
        frame_index = 0

        try:
            with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
                while True:
                    decode_ok, frame = capture.read()
                    if not decode_ok:
                        break
                    timestamp_ns = timestamps[frame_index] if frame_index < len(timestamps) else 0
                    inference_ms = max(previous_inference_ms + 1, round(timestamp_ns / 1_000_000))
                    previous_inference_ms = inference_ms
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect_for_video(image, inference_ms)
                    height, width = frame.shape[:2]

                    detection_inputs: list[tuple[float, float, str, float]] = []
                    categories: list[tuple[str, float]] = []
                    for detection_index, landmarks in enumerate(result.hand_landmarks):
                        side, confidence = _category(result, detection_index)
                        categories.append((side, confidence))
                        detection_inputs.append(
                            (float(landmarks[0].x), float(landmarks[0].y), side, confidence)
                        )
                    assignments = assigner.assign(frame_index, detection_inputs)

                    stable_sides: list[str] = []
                    for detection_index, landmarks in enumerate(result.hand_landmarks):
                        track_id, stable_side = assignments[detection_index]
                        _, side_confidence = categories[detection_index]
                        stable_sides.append(stable_side)
                        normalized = [(float(point.x), float(point.y)) for point in landmarks]
                        landmarks_px = [[x * width, y * height] for x, y in normalized]
                        xs = [point[0] for point in landmarks_px]
                        ys = [point[1] for point in landmarks_px]
                        inside = [0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in normalized]
                        world = (
                            result.hand_world_landmarks[detection_index]
                            if detection_index < len(result.hand_world_landmarks)
                            else []
                        )
                        relative_3d = (
                            [[float(point.x), float(point.y), float(point.z)] for point in world]
                            if len(world) == 21
                            else None
                        )
                        hand_rows.append(
                            {
                                "frame_index": frame_index,
                                "timestamp_ns": timestamp_ns,
                                "detection_index": detection_index,
                                "track_id": track_id,
                                "side": stable_side,
                                "side_confidence": side_confidence,
                                "bbox_xyxy_px": [min(xs), min(ys), max(xs), max(ys)],
                                "landmarks_2d_px": landmarks_px,
                                "landmarks_relative_3d": relative_3d,
                                # MediaPipe does not expose per-joint confidence. Preserve
                                # the 21-joint shape while keeping every value explicitly null.
                                "landmark_confidence": [None] * 21,
                                "visible_fraction": sum(inside) / 21.0,
                                "occluded": None,
                                "truncated_by_frame": not all(inside),
                                "observation_state": "observed",
                                "provenance": ProvenanceClass.OFFLINE_ESTIMATED.value,
                                "epistemic_status": EpistemicStatus.ESTIMATE.value,
                                "processor_id": self.processor_id,
                                "model_sha256": model_hash,
                            }
                        )
                    frame_rows.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_ns": timestamp_ns,
                            "inference_timestamp_ms": inference_ms,
                            "decode_ok": True,
                            "hand_count": len(result.hand_landmarks),
                            "left_count": stable_sides.count("left"),
                            "right_count": stable_sides.count("right"),
                            "unknown_count": stable_sides.count("unknown"),
                            "processor_id": self.processor_id,
                            "model_sha256": model_hash,
                        }
                    )
                    frame_index += 1
        finally:
            capture.release()

        for missing_index in range(frame_index, len(timestamps)):
            frame_rows.append(
                {
                    "frame_index": missing_index,
                    "timestamp_ns": timestamps[missing_index],
                    "inference_timestamp_ms": None,
                    "decode_ok": False,
                    "hand_count": 0,
                    "left_count": 0,
                    "right_count": 0,
                    "unknown_count": 0,
                    "processor_id": self.processor_id,
                    "model_sha256": model_hash,
                }
            )

        hands_path = workspace.derived / "hands.parquet"
        hand_frames_path = workspace.derived / "hand_frames.parquet"
        pq.write_table(pa.Table.from_pylist(hand_rows, schema=HANDS_SCHEMA), hands_path, compression="zstd")
        pq.write_table(
            pa.Table.from_pylist(frame_rows, schema=HAND_FRAMES_SCHEMA),
            hand_frames_path,
            compression="zstd",
        )
        return ProcessorResult(
            outputs=[hands_path, hand_frames_path],
            metrics={
                "frames_processed": frame_index,
                "hand_observations": len(hand_rows),
                "frames_with_hands": sum(row["hand_count"] > 0 for row in frame_rows),
                "unique_track_ids": len({row["track_id"] for row in hand_rows}),
                "mediapipe_version": mp.__version__,
                "model_sha256": model_hash,
                "model_source": model_source,
                "per_joint_confidence": "not_exposed_by_engine; stored as null",
                "relative_3d_note": "model-relative hand coordinates; not camera/world metric truth",
            },
        )
