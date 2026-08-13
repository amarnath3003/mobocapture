from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mobocapture.io import write_json
from mobocapture.models import EpistemicStatus, ProvenanceClass
from mobocapture.processors.base import Processor, ProcessorResult
from mobocapture.schemas import QUALITY_SCHEMA
from mobocapture.session import SessionWorkspace


def _motion_category(delta: float | None) -> str | None:
    if delta is None:
        return None
    if delta < 1.0:
        return "static"
    if delta < 4.0:
        return "slow"
    if delta < 12.0:
        return "moderate"
    return "rapid_or_cut"


class VideoQualityProcessor(Processor):
    processor_id = "video_quality"

    def process(self, workspace: SessionWorkspace) -> ProcessorResult:
        frame_table = pq.read_table(workspace.derived / "frame_index.parquet")
        timestamps = frame_table.column("timestamp_ns").to_pylist()
        capture = cv2.VideoCapture(str(workspace.raw_video))
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the copied input video")

        rows: list[dict[str, Any]] = []
        previous_gray: np.ndarray | None = None
        previous_frame: np.ndarray | None = None
        index = 0
        while True:
            decode_ok, frame = capture.read()
            if not decode_ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray)) / 255.0
            underexposed = float(np.mean(gray <= 16))
            overexposed = float(np.mean(gray >= 240))
            laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            # This is a review-prioritization heuristic, not a calibrated probability.
            blur_likelihood = float(1.0 / (1.0 + laplacian_variance / 100.0))
            exact_duplicate = previous_frame is not None and np.array_equal(frame, previous_frame)
            delta = (
                float(cv2.absdiff(gray, previous_gray).mean())
                if previous_gray is not None
                else None
            )
            near_duplicate = delta is not None and delta < 0.75
            timestamp = timestamps[index] if index < len(timestamps) else 0
            rows.append(
                {
                    "frame_index": index,
                    "timestamp_ns": timestamp,
                    "decode_ok": True,
                    "brightness_mean": brightness,
                    "underexposed_fraction": underexposed,
                    "overexposed_fraction": overexposed,
                    "laplacian_variance": laplacian_variance,
                    "blur_likelihood": blur_likelihood,
                    "frame_delta_mean": delta,
                    "exact_duplicate": exact_duplicate,
                    "near_duplicate": near_duplicate,
                    "camera_motion_category": _motion_category(delta),
                    "pixel_stats_provenance": ProvenanceClass.OFFLINE_ESTIMATED.value,
                    "pixel_stats_status": EpistemicStatus.DETERMINISTIC.value,
                    "blur_status": EpistemicStatus.ESTIMATE.value,
                    "processor_id": self.processor_id,
                }
            )
            previous_gray = gray
            previous_frame = frame.copy()
            index += 1
        capture.release()

        for missing_index in range(index, len(timestamps)):
            rows.append(
                {
                    "frame_index": missing_index,
                    "timestamp_ns": timestamps[missing_index],
                    "decode_ok": False,
                    "brightness_mean": None,
                    "underexposed_fraction": None,
                    "overexposed_fraction": None,
                    "laplacian_variance": None,
                    "blur_likelihood": None,
                    "frame_delta_mean": None,
                    "exact_duplicate": False,
                    "near_duplicate": False,
                    "camera_motion_category": None,
                    "pixel_stats_provenance": ProvenanceClass.OFFLINE_ESTIMATED.value,
                    "pixel_stats_status": EpistemicStatus.DETERMINISTIC.value,
                    "blur_status": EpistemicStatus.ESTIMATE.value,
                    "processor_id": self.processor_id,
                }
            )

        quality_path = workspace.derived / "quality.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=QUALITY_SCHEMA),
            quality_path,
            compression="zstd",
        )
        decoded_rows = [row for row in rows if row["decode_ok"]]
        summary = {
            "schema_version": "0.1.0",
            "processor_id": self.processor_id,
            "probed_frames": len(timestamps),
            "decoded_frames": len(decoded_rows),
            "decode_failures": len(rows) - len(decoded_rows),
            "exact_duplicate_frames": sum(row["exact_duplicate"] for row in rows),
            "near_duplicate_frames": sum(row["near_duplicate"] for row in rows),
            "mean_brightness": (
                float(np.mean([row["brightness_mean"] for row in decoded_rows]))
                if decoded_rows
                else None
            ),
            "mean_blur_likelihood": (
                float(np.mean([row["blur_likelihood"] for row in decoded_rows]))
                if decoded_rows
                else None
            ),
            "notes": {
                "blur_likelihood": "Uncalibrated review-prioritization heuristic",
                "camera_motion_category": "Pixel-difference category; rapid motion and cuts are not separated yet",
            },
        }
        summary_path = workspace.derived / "quality_summary.json"
        write_json(summary_path, summary)
        return ProcessorResult(
            outputs=[quality_path, summary_path],
            metrics={
                "probed_frames": len(timestamps),
                "decoded_frames": len(decoded_rows),
                "decode_failures": len(rows) - len(decoded_rows),
                "opencv_version": cv2.__version__,
            },
        )
