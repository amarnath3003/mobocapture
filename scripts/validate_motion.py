"""Validate point tracks and dense flow against OpenCV's public motion sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from mobocapture.pipeline import process_video


SAMPLE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"
SAMPLE_SHA256 = "45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf"


def download_verified(destination: Path) -> None:
    request = urllib.request.Request(
        SAMPLE_URL, headers={"User-Agent": "MoboCapture/0.1 validator"}
    )
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)
    actual = digest.hexdigest()
    if actual != SAMPLE_SHA256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Sample hash mismatch: expected {SAMPLE_SHA256}, got {actual}")


def run(output_root: Path) -> dict:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sample = output_root / "opencv-vtest.avi"
    if not sample.is_file():
        download_verified(sample)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required")
    clip = output_root / "motion-clip.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(sample),
            "-frames:v",
            "30",
            "-vf",
            "scale=640:-2",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    session = output_root / "session"
    if session.exists():
        raise FileExistsError(f"Validation session already exists: {session}")
    workspace = process_video(clip, session, module_ids=["motion_tracking"])

    points = pq.read_table(workspace.derived / "point_tracks.parquet").to_pylist()
    point_frames = pq.read_table(workspace.derived / "point_frames.parquet").to_pylist()
    flow_rows = pq.read_table(workspace.derived / "optical_flow.parquet").to_pylist()
    track_lengths = Counter(row["track_id"] for row in points)
    flow_files_valid = True
    for row in flow_rows:
        with np.load(workspace.root / row["flow_path"]) as payload:
            flow = payload["flow_xy_px"]
            flow_files_valid = bool(
                flow_files_valid
                and flow.shape == (row["height"], row["width"], 2)
                and np.isfinite(flow).all()
            )
    assertions = {
        "thirty_frames_processed": len(point_frames) == 30,
        "twenty_nine_dense_flow_pairs": len(flow_rows) == 29,
        "dense_arrays_valid": bool(flow_rows) and flow_files_valid,
        "nonzero_motion_measured": max(
            (row["p95_magnitude_px"] for row in flow_rows), default=0
        ) > 0.5,
        "persistent_point_tracks": bool(
            sum(length >= 10 for length in track_lengths.values()) >= 20
        ),
        "tracked_rows_have_displacement": all(
            row["displacement_xy_px"] is not None
            for row in points
            if row["observation_state"] == "tracked"
        ),
        "session_complete": workspace.manifest.status == "complete",
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)

    source_capture = cv2.VideoCapture(str(clip))
    source_capture.set(cv2.CAP_PROP_POS_FRAMES, 15)
    source_ok, source_frame = source_capture.read()
    source_capture.release()
    overlay_capture = cv2.VideoCapture(str(workspace.review / "overlay.mp4"))
    overlay_capture.set(cv2.CAP_PROP_POS_FRAMES, 15)
    overlay_ok, overlay = overlay_capture.read()
    overlay_capture.release()
    if not source_ok or not overlay_ok:
        raise RuntimeError("Could not decode validation comparison frames")
    source_frame = cv2.resize(source_frame, (overlay.shape[1], overlay.shape[0]))
    comparison = cv2.hconcat([source_frame, overlay])
    for x, label in (
        (0, "ORIGINAL OPENCV VIDEO"),
        (overlay.shape[1], "MOBOCAPTURE MOTION OUTPUT"),
    ):
        cv2.rectangle(
            comparison,
            (x, comparison.shape[0] - 44),
            (x + overlay.shape[1], comparison.shape[0]),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            comparison,
            label,
            (x + 16, comparison.shape[0] - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    comparison_path = output_root / "side-by-side.png"
    cv2.imwrite(str(comparison_path), comparison)

    report = {
        "module": "motion_tracking",
        "sample_url": SAMPLE_URL,
        "sample_sha256": SAMPLE_SHA256,
        "assertions": assertions,
        "frames": len(point_frames),
        "point_observations": len(points),
        "unique_point_tracks": len(track_lengths),
        "tracks_10_or_more_frames": sum(length >= 10 for length in track_lengths.values()),
        "maximum_track_length_frames": max(track_lengths.values(), default=0),
        "flow_pairs": len(flow_rows),
        "mean_flow_magnitude_px": float(
            np.mean([row["mean_magnitude_px"] for row in flow_rows])
        ),
        "comparison": str(comparison_path),
        "session": str(workspace.root),
    }
    (output_root / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("validation-output/motion"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
