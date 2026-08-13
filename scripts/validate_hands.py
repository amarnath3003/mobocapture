"""Run the Hands & Fingers module against Google's official internet sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

import cv2
import pyarrow.parquet as pq

from mobocapture.pipeline import process_video


SAMPLE_URL = "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg"
SAMPLE_SHA256 = "70cbeb38e198c9862202e0979c21a99b40ca980d3e7b250176c85b1636a40f12"


def download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "MoboCapture/0.1 validator"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Sample hash mismatch: expected {expected_sha256}, got {actual}")


def run(output_root: Path) -> dict:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sample = output_root / "woman_hands.jpg"
    if not sample.is_file():
        download_verified(SAMPLE_URL, sample, SAMPLE_SHA256)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required")
    video = output_root / "woman_hands.mp4"
    subprocess.run(
        [
            ffmpeg, "-y", "-v", "error", "-loop", "1", "-i", str(sample),
            "-t", "2", "-r", "15", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )
    session = output_root / "session"
    if session.exists():
        raise FileExistsError(f"Validation session already exists: {session}")
    workspace = process_video(video, session, module_ids=["hands_fingers"])

    hands = pq.read_table(workspace.derived / "hands.parquet").to_pylist()
    frames = pq.read_table(workspace.derived / "hand_frames.parquet").to_pylist()
    assertions = {
        "30_frames_processed": len(frames) == 30,
        "two_hands_each_frame": bool(frames) and min(row["hand_count"] for row in frames) == 2,
        "21_landmarks_each_observation": bool(hands)
        and all(len(row["landmarks_2d_px"]) == 21 for row in hands),
        "left_and_right_detected": {row["side"] for row in hands} == {"left", "right"},
        "two_persistent_tracks": len({row["track_id"] for row in hands}) == 2,
        "session_complete": workspace.manifest.status == "complete",
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)

    original = cv2.imread(str(sample))
    capture = cv2.VideoCapture(str(workspace.review / "overlay.mp4"))
    capture.set(cv2.CAP_PROP_POS_MSEC, 400)
    ok, overlay = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Could not decode the validation overlay")
    original = cv2.resize(original, (overlay.shape[1], overlay.shape[0]))
    comparison = cv2.hconcat([original, overlay])
    for x, label in ((0, "ORIGINAL INTERNET SAMPLE"), (overlay.shape[1], "MOBOCAPTURE HANDS OUTPUT")):
        cv2.rectangle(
            comparison,
            (x, comparison.shape[0] - 45),
            (x + overlay.shape[1], comparison.shape[0]),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            comparison,
            label,
            (x + 18, comparison.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    comparison_path = output_root / "side-by-side.png"
    cv2.imwrite(str(comparison_path), comparison)

    report = {
        "module": "hands_fingers",
        "sample_url": SAMPLE_URL,
        "sample_sha256": SAMPLE_SHA256,
        "assertions": assertions,
        "frames": len(frames),
        "hand_observations": len(hands),
        "track_ids": sorted({row["track_id"] for row in hands}),
        "sides": sorted({row["side"] for row in hands}),
        "comparison": str(comparison_path),
        "session": str(workspace.root),
    }
    (output_root / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("validation-output/hands"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
