from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mobocapture.models import EpistemicStatus, ProvenanceClass
from mobocapture.processors.base import Processor, ProcessorResult
from mobocapture.schemas import OPTICAL_FLOW_SCHEMA, POINT_FRAMES_SCHEMA, POINT_TRACKS_SCHEMA
from mobocapture.session import SessionWorkspace


POINT_ALGORITHM = "Shi-Tomasi corners + pyramidal Lucas-Kanade + forward/backward check"
FLOW_ALGORITHM = "Farneback dense optical flow (OpenCV)"


def _save_npz_lossless_fast(path: Path, **arrays: np.ndarray) -> None:
    """Write an np.load-compatible NPZ with fast, still-lossless compression."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for name, array in arrays.items():
            with archive.open(f"{name}.npy", mode="w", force_zip64=True) as stream:
                np.lib.format.write_array(
                    stream, np.asanyarray(array), allow_pickle=False
                )
    os.replace(temporary, path)


def _new_features(
    gray: np.ndarray,
    existing: np.ndarray | None,
    maximum_points: int,
) -> np.ndarray:
    existing_count = 0 if existing is None else len(existing)
    remaining = maximum_points - existing_count
    if remaining <= 0:
        return np.empty((0, 1, 2), dtype=np.float32)
    mask = np.full(gray.shape, 255, dtype=np.uint8)
    if existing is not None:
        for x, y in existing.reshape(-1, 2):
            cv2.circle(mask, (int(round(x)), int(round(y))), 9, 0, -1)
    features = cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        maxCorners=remaining,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
        useHarrisDetector=False,
    )
    if features is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return features.astype(np.float32)


class PointTrackerProcessor(Processor):
    processor_id = "point_tracker"

    def process(self, workspace: SessionWorkspace) -> ProcessorResult:
        frame_table = pq.read_table(workspace.derived / "frame_index.parquet")
        timestamps = frame_table.column("timestamp_ns").to_pylist()
        capture = cv2.VideoCapture(str(workspace.raw_video))
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the copied video for point tracking")

        maximum_points = 300
        minimum_points = 180
        lk_parameters = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }
        rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        next_track_id = 1
        decoded, first_frame = capture.read()
        if not decoded:
            capture.release()
            raise RuntimeError("Cannot decode the first video frame for point tracking")
        previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        points = _new_features(previous_gray, None, maximum_points)
        track_ids = [f"point-{index:06d}" for index in range(1, len(points) + 1)]
        next_track_id += len(points)
        first_timestamp = timestamps[0] if timestamps else 0
        for track_id, point in zip(track_ids, points.reshape(-1, 2)):
            rows.append(
                {
                    "frame_index": 0,
                    "timestamp_ns": first_timestamp,
                    "track_id": track_id,
                    "xy_px": point.tolist(),
                    "previous_xy_px": None,
                    "displacement_xy_px": None,
                    "velocity_xy_px_s": None,
                    "tracking_error": None,
                    "forward_backward_error_px": None,
                    "observation_state": "seeded",
                    "provenance": ProvenanceClass.OFFLINE_ESTIMATED.value,
                    "epistemic_status": EpistemicStatus.ESTIMATE.value,
                    "processor_id": self.processor_id,
                    "algorithm": POINT_ALGORITHM,
                }
            )
        frame_rows.append(
            {
                "frame_index": 0,
                "timestamp_ns": first_timestamp,
                "decode_ok": True,
                "visible_point_count": len(points),
                "tracked_point_count": 0,
                "seeded_point_count": len(points),
                "lost_point_count": 0,
                "processor_id": self.processor_id,
                "algorithm": POINT_ALGORITHM,
            }
        )

        frame_index = 1
        try:
            while True:
                decoded, frame = capture.read()
                if not decoded:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                timestamp_ns = timestamps[frame_index] if frame_index < len(timestamps) else 0
                previous_timestamp = (
                    timestamps[frame_index - 1] if frame_index - 1 < len(timestamps) else 0
                )
                delta_seconds = (timestamp_ns - previous_timestamp) / 1_000_000_000
                tracked_points: list[np.ndarray] = []
                tracked_ids: list[str] = []
                tracked_errors: list[float] = []
                fb_errors: list[float] = []
                old_points: list[np.ndarray] = []
                lost_count = len(points)
                if len(points):
                    next_points, status, error = cv2.calcOpticalFlowPyrLK(
                        previous_gray, gray, points, None, **lk_parameters
                    )
                    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
                        gray, previous_gray, next_points, None, **lk_parameters
                    )
                    forward = next_points.reshape(-1, 2)
                    backward = back_points.reshape(-1, 2)
                    original = points.reshape(-1, 2)
                    status_values = status.reshape(-1).astype(bool)
                    back_values = back_status.reshape(-1).astype(bool)
                    error_values = error.reshape(-1)
                    height, width = gray.shape
                    for index, (old, new, back) in enumerate(
                        zip(original, forward, backward)
                    ):
                        fb_error = float(np.linalg.norm(old - back))
                        inside = 0 <= new[0] < width and 0 <= new[1] < height
                        good = (
                            status_values[index]
                            and back_values[index]
                            and inside
                            and math_is_finite(fb_error)
                            and fb_error <= 1.5
                            and float(error_values[index]) <= 40.0
                        )
                        if good:
                            old_points.append(old)
                            tracked_points.append(new)
                            tracked_ids.append(track_ids[index])
                            tracked_errors.append(float(error_values[index]))
                            fb_errors.append(fb_error)
                    lost_count = len(points) - len(tracked_points)

                for track_id, old, new, error_value, fb_error in zip(
                    tracked_ids,
                    old_points,
                    tracked_points,
                    tracked_errors,
                    fb_errors,
                ):
                    displacement = new - old
                    velocity = (
                        (displacement / delta_seconds).tolist() if delta_seconds > 0 else None
                    )
                    rows.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_ns": timestamp_ns,
                            "track_id": track_id,
                            "xy_px": new.tolist(),
                            "previous_xy_px": old.tolist(),
                            "displacement_xy_px": displacement.tolist(),
                            "velocity_xy_px_s": velocity,
                            "tracking_error": error_value,
                            "forward_backward_error_px": fb_error,
                            "observation_state": "tracked",
                            "provenance": ProvenanceClass.OFFLINE_ESTIMATED.value,
                            "epistemic_status": EpistemicStatus.ESTIMATE.value,
                            "processor_id": self.processor_id,
                            "algorithm": POINT_ALGORITHM,
                        }
                    )

                current = (
                    np.asarray(tracked_points, dtype=np.float32).reshape(-1, 1, 2)
                    if tracked_points
                    else np.empty((0, 1, 2), dtype=np.float32)
                )
                new_points = (
                    _new_features(gray, current, maximum_points)
                    if len(current) < minimum_points
                    else np.empty((0, 1, 2), dtype=np.float32)
                )
                new_ids = []
                for point in new_points.reshape(-1, 2):
                    track_id = f"point-{next_track_id:06d}"
                    next_track_id += 1
                    new_ids.append(track_id)
                    rows.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_ns": timestamp_ns,
                            "track_id": track_id,
                            "xy_px": point.tolist(),
                            "previous_xy_px": None,
                            "displacement_xy_px": None,
                            "velocity_xy_px_s": None,
                            "tracking_error": None,
                            "forward_backward_error_px": None,
                            "observation_state": "seeded",
                            "provenance": ProvenanceClass.OFFLINE_ESTIMATED.value,
                            "epistemic_status": EpistemicStatus.ESTIMATE.value,
                            "processor_id": self.processor_id,
                            "algorithm": POINT_ALGORITHM,
                        }
                    )
                points = (
                    np.concatenate([current, new_points], axis=0)
                    if len(new_points)
                    else current
                )
                track_ids = tracked_ids + new_ids
                frame_rows.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_ns": timestamp_ns,
                        "decode_ok": True,
                        "visible_point_count": len(points),
                        "tracked_point_count": len(tracked_points),
                        "seeded_point_count": len(new_points),
                        "lost_point_count": lost_count,
                        "processor_id": self.processor_id,
                        "algorithm": POINT_ALGORITHM,
                    }
                )
                previous_gray = gray
                frame_index += 1
        finally:
            capture.release()

        for missing_index in range(frame_index, len(timestamps)):
            frame_rows.append(
                {
                    "frame_index": missing_index,
                    "timestamp_ns": timestamps[missing_index],
                    "decode_ok": False,
                    "visible_point_count": 0,
                    "tracked_point_count": 0,
                    "seeded_point_count": 0,
                    "lost_point_count": 0,
                    "processor_id": self.processor_id,
                    "algorithm": POINT_ALGORITHM,
                }
            )

        tracks_path = workspace.derived / "point_tracks.parquet"
        frames_path = workspace.derived / "point_frames.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=POINT_TRACKS_SCHEMA),
            tracks_path,
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(frame_rows, schema=POINT_FRAMES_SCHEMA),
            frames_path,
            compression="zstd",
        )
        observation_counts: dict[str, int] = {}
        for row in rows:
            observation_counts[row["track_id"]] = observation_counts.get(row["track_id"], 0) + 1
        return ProcessorResult(
            outputs=[tracks_path, frames_path],
            metrics={
                "frames_processed": frame_index,
                "point_observations": len(rows),
                "unique_track_ids": len(observation_counts),
                "tracks_observed_5_or_more_frames": sum(
                    count >= 5 for count in observation_counts.values()
                ),
                "maximum_track_length_frames": max(observation_counts.values(), default=0),
                "algorithm": POINT_ALGORITHM,
                "maximum_points_per_frame": maximum_points,
            },
        )


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(value))


class OpticalFlowProcessor(Processor):
    processor_id = "optical_flow"

    def process(self, workspace: SessionWorkspace) -> ProcessorResult:
        frame_table = pq.read_table(workspace.derived / "frame_index.parquet")
        timestamps = frame_table.column("timestamp_ns").to_pylist()
        capture = cv2.VideoCapture(str(workspace.raw_video))
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the copied video for optical flow")
        decoded, previous_frame = capture.read()
        if not decoded:
            capture.release()
            raise RuntimeError("Cannot decode the first video frame for optical flow")
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        flow_directory = workspace.derived / "optical_flow"
        flow_directory.mkdir(parents=True, exist_ok=True)
        parameters = {
            "pyr_scale": 0.5,
            "levels": 4,
            "winsize": 21,
            "iterations": 4,
            "poly_n": 7,
            "poly_sigma": 1.5,
            "flags": 0,
        }
        rows: list[dict[str, Any]] = []
        outputs: list[Path] = []
        target_index = 1
        try:
            while True:
                decoded, frame = capture.read()
                if not decoded:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    previous_gray,
                    gray,
                    None,
                    parameters["pyr_scale"],
                    parameters["levels"],
                    parameters["winsize"],
                    parameters["iterations"],
                    parameters["poly_n"],
                    parameters["poly_sigma"],
                    parameters["flags"],
                )
                finite = np.isfinite(flow).all(axis=2)
                magnitude = np.linalg.norm(flow, axis=2)
                valid_magnitude = magnitude[finite]
                destination = flow_directory / f"frame_{target_index:08d}.npz"
                _save_npz_lossless_fast(
                    destination, flow_xy_px=flow.astype(np.float16)
                )
                outputs.append(destination)
                height, width = gray.shape
                source_timestamp = timestamps[target_index - 1] if target_index - 1 < len(timestamps) else 0
                target_timestamp = timestamps[target_index] if target_index < len(timestamps) else 0
                rows.append(
                    {
                        "source_frame_index": target_index - 1,
                        "target_frame_index": target_index,
                        "source_timestamp_ns": source_timestamp,
                        "target_timestamp_ns": target_timestamp,
                        "delta_time_ns": max(0, target_timestamp - source_timestamp),
                        "flow_path": destination.relative_to(workspace.root).as_posix(),
                        "width": width,
                        "height": height,
                        "component_order": "x_right_px, y_down_px",
                        "dtype": "float16",
                        "valid_fraction": float(finite.mean()),
                        "mean_magnitude_px": float(valid_magnitude.mean()),
                        "median_magnitude_px": float(np.median(valid_magnitude)),
                        "p95_magnitude_px": float(np.percentile(valid_magnitude, 95)),
                        "maximum_magnitude_px": float(valid_magnitude.max()),
                        "provenance": ProvenanceClass.OFFLINE_ESTIMATED.value,
                        "epistemic_status": EpistemicStatus.ESTIMATE.value,
                        "processor_id": self.processor_id,
                        "algorithm": FLOW_ALGORITHM,
                        "parameters_json": json.dumps(parameters, sort_keys=True),
                    }
                )
                previous_gray = gray
                target_index += 1
        finally:
            capture.release()

        index_path = workspace.derived / "optical_flow.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=OPTICAL_FLOW_SCHEMA),
            index_path,
            compression="zstd",
        )
        outputs.insert(0, index_path)
        return ProcessorResult(
            outputs=outputs,
            metrics={
                "frame_pairs_processed": len(rows),
                "dense_flow_arrays": len(rows),
                "algorithm": FLOW_ALGORITHM,
                "storage": (
                    "lossless fast-compressed NPZ containing float16 "
                    "flow_xy_px [height,width,2]"
                ),
                "mean_pair_magnitude_px": float(
                    np.mean([row["mean_magnitude_px"] for row in rows])
                ) if rows else 0.0,
            },
        )
