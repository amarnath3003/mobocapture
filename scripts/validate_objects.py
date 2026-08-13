"""Validate boxes, masks, tracks, dataset rows, and overlay on a public SAM2 fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

import cv2
import pyarrow.parquet as pq

from mobocapture.pipeline import process_video


SAMPLE_URL = (
    "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/"
    "resolve/main/truck.jpg"
)
SAMPLE_SHA256 = "941715e721c8864324a1425b445ea4dde0498b995c45ddce0141a58971c6ff99"


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
    sample = output_root / "truck.jpg"
    if not sample.is_file():
        download_verified(sample)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required")
    video = output_root / "truck.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-loop",
            "1",
            "-i",
            str(sample),
            "-frames:v",
            "3",
            "-r",
            "3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    session = output_root / "session"
    if session.exists():
        raise FileExistsError(f"Validation session already exists: {session}")
    workspace = process_video(
        video,
        session,
        module_ids=["objects"],
        options={"objects": {"concepts": ["truck", "wheel"]}},
    )

    detections = pq.read_table(workspace.derived / "object_detections.parquet").to_pylist()
    regions = pq.read_table(workspace.derived / "regions.parquet").to_pylist()
    frames = pq.read_table(workspace.derived / "object_frames.parquet").to_pylist()
    labels = {row["label"] for row in regions}
    tracks_by_label = {
        label: {row["track_id"] for row in regions if row["label"] == label}
        for label in labels
    }
    assertions = {
        "three_frames_processed": len(frames) == 3,
        "detections_and_masks_match": len(detections) == len(regions) and len(regions) > 0,
        "truck_and_wheel_found": {"truck", "wheel"}.issubset(labels),
        "all_masks_nonempty": bool(regions) and all(row["mask_area_px"] > 0 for row in regions),
        "all_masks_have_rle": bool(regions)
        and all(sum(row["mask_rle_counts"]) == 1800 * 1200 for row in regions),
        "tracks_persist": all(
            len(track_ids) <= sum(row["label"] == label for row in regions) / 3
            for label, track_ids in tracks_by_label.items()
        ),
        "session_complete": workspace.manifest.status == "complete",
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)

    original = cv2.imread(str(sample))
    capture = cv2.VideoCapture(str(workspace.review / "overlay.mp4"))
    ok, overlay = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Could not decode the validation overlay")
    original = cv2.resize(original, (overlay.shape[1], overlay.shape[0]))
    comparison = cv2.hconcat([original, overlay])
    for x, label in (
        (0, "ORIGINAL PUBLIC FIXTURE"),
        (overlay.shape[1], "MOBOCAPTURE OBJECT OUTPUT"),
    ):
        cv2.rectangle(
            comparison,
            (x, comparison.shape[0] - 48),
            (x + overlay.shape[1], comparison.shape[0]),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            comparison,
            label,
            (x + 18, comparison.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    comparison_path = output_root / "side-by-side.png"
    cv2.imwrite(str(comparison_path), comparison)

    report = {
        "module": "objects",
        "sample_url": SAMPLE_URL,
        "sample_sha256": SAMPLE_SHA256,
        "assertions": assertions,
        "frames": len(frames),
        "detections": len(detections),
        "regions": len(regions),
        "labels": sorted(labels),
        "track_ids_by_label": {
            label: sorted(track_ids) for label, track_ids in tracks_by_label.items()
        },
        "mask_area_range_px": [
            min(row["mask_area_px"] for row in regions),
            max(row["mask_area_px"] for row in regions),
        ],
        "comparison": str(comparison_path),
        "session": str(workspace.root),
    }
    (output_root / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("validation-output/objects"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
