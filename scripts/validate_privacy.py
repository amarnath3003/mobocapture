"""Validate privacy candidates, tracking, review overlay, and redacted derivative."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import pyarrow.parquet as pq
import requests

from mobocapture.pipeline import process_video


SAMPLE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/"
    "Female_with_computer_laptop.jpg/960px-Female_with_computer_laptop.jpg"
)
SAMPLE_PAGE = "https://commons.wikimedia.org/wiki/File:Female_with_computer_laptop.jpg"
SAMPLE_SHA256 = "5b3aa311eb9546ceeaf4296c778b3849741cb0f69cfc04b83f473442022a5e90"


def download_verified(destination: Path) -> None:
    digest = hashlib.sha256()
    with requests.get(
        SAMPLE_URL,
        headers={"User-Agent": "MoboCapture/0.1 validator"},
        timeout=60,
        stream=True,
    ) as response, destination.open("wb") as output:
        response.raise_for_status()
        for chunk in response.iter_content(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)
    actual = digest.hexdigest()
    if actual != SAMPLE_SHA256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Sample hash mismatch: expected {SAMPLE_SHA256}, got {actual}")


def run(output_root: Path) -> dict:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sample = output_root / "person-laptop-documents.jpg"
    if not sample.is_file():
        download_verified(sample)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required")
    video = output_root / "person-laptop-documents.mp4"
    subprocess.run(
        [
            ffmpeg, "-y", "-v", "error", "-loop", "1", "-i", str(sample),
            "-frames:v", "3", "-r", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video),
        ],
        check=True,
    )
    session = output_root / "session"
    if session.exists():
        raise FileExistsError(f"Validation session already exists: {session}")
    workspace = process_video(video, session, module_ids=["privacy_redaction"])
    regions = pq.read_table(workspace.derived / "privacy_regions.parquet").to_pylist()
    frames = pq.read_table(workspace.derived / "privacy_frames.parquet").to_pylist()
    categories = {row["category"] for row in regions}
    redacted_path = workspace.review / "redacted.mp4"
    assertions = {
        "three_frames_scanned": len(frames) == 3,
        "face_detected_every_frame": all(row["face_count"] >= 1 for row in frames),
        "screen_and_document_detected": {"computer screen", "document"}.issubset(categories),
        "privacy_tracks_persist": all(
            len({row["track_id"] for row in regions if row["category"] == category})
            <= sum(row["category"] == category for row in regions) / 3 + 1
            for category in {"face", "computer screen", "document"}
        ),
        "all_candidates_require_review": bool(regions)
        and all(row["review_required"] for row in regions),
        "redacted_derivative_exists": redacted_path.is_file() and redacted_path.stat().st_size > 0,
        "original_preserved": workspace.raw_video.is_file(),
        "session_complete": workspace.manifest.status == "complete",
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)

    original = cv2.imread(str(sample))
    overlay_capture = cv2.VideoCapture(str(workspace.review / "overlay.mp4"))
    overlay_ok, overlay = overlay_capture.read()
    overlay_capture.release()
    redacted_capture = cv2.VideoCapture(str(redacted_path))
    redacted_ok, redacted = redacted_capture.read()
    redacted_capture.release()
    if not overlay_ok or not redacted_ok:
        raise RuntimeError("Could not decode privacy validation outputs")
    original = cv2.resize(original, (overlay.shape[1], overlay.shape[0]))
    redacted = cv2.resize(redacted, (overlay.shape[1], overlay.shape[0]))
    comparison = cv2.hconcat([original, overlay, redacted])
    labels = ("ORIGINAL", "PRIVACY REVIEW OVERLAY", "REDACTED DERIVATIVE")
    for index, label in enumerate(labels):
        x = index * overlay.shape[1]
        cv2.rectangle(
            comparison, (x, comparison.shape[0] - 44),
            (x + overlay.shape[1], comparison.shape[0]), (0, 0, 0), -1,
        )
        cv2.putText(
            comparison, label, (x + 16, comparison.shape[0] - 13),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
    comparison_path = output_root / "side-by-side.png"
    cv2.imwrite(str(comparison_path), comparison)
    report = {
        "module": "privacy_redaction",
        "sample_url": SAMPLE_URL,
        "sample_page": SAMPLE_PAGE,
        "sample_license": "Public domain; Steve Hillebrand, U.S. Fish and Wildlife Service",
        "sample_sha256": SAMPLE_SHA256,
        "assertions": assertions,
        "frames": len(frames),
        "privacy_regions": len(regions),
        "categories": sorted(categories),
        "comparison": str(comparison_path),
        "redacted_video": str(redacted_path),
        "session": str(workspace.root),
    }
    (output_root / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("validation-output/privacy"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
