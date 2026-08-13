"""Validate evidence-linked hand/object hypotheses on a public Wikimedia image."""

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
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/"
    "DFC_0816_A_close-up_of_a_hand_holding_a_plastic_cup_while_a_chocolatey_"
    "smoothie_is_poured_over_ice_from_a_blender.jpg/"
    "960px-DFC_0816_A_close-up_of_a_hand_holding_a_plastic_cup_while_a_"
    "chocolatey_smoothie_is_poured_over_ice_from_a_blender.jpg"
)
SAMPLE_PAGE = (
    "https://commons.wikimedia.org/wiki/File:DFC_0816_A_close-up_of_a_hand_"
    "holding_a_plastic_cup_while_a_chocolatey_smoothie_is_poured_over_ice_"
    "from_a_blender.jpg"
)
SAMPLE_SHA256 = "005cb72de1ad323205868cab04e61e7e2d94df99fbecb298929696cba63c537d"


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
    sample = output_root / "hand-holding-cup.jpg"
    if not sample.is_file():
        download_verified(sample)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required")
    video = output_root / "hand-holding-cup.mp4"
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
        module_ids=["hand_object_interactions"],
        options={"objects": {"concepts": ["plastic cup", "cup", "glass"]}},
    )

    hands = pq.read_table(workspace.derived / "hands.parquet").to_pylist()
    regions = pq.read_table(workspace.derived / "regions.parquet").to_pylist()
    interactions = pq.read_table(workspace.derived / "interactions.parquet").to_pylist()
    assigned = [row for row in interactions if row["assigned_to_hand"]]
    assertions = {
        "hand_detected_every_frame": len(hands) == 3
        and len({row["track_id"] for row in hands}) == 1,
        "cup_segmented_every_frame": sum(
            row["label"] == "plastic cup" for row in regions
        ) >= 3,
        "interaction_assigned_every_frame": len(assigned) == 3,
        "near_cup_evidence": bool(assigned)
        and max(row["minimum_fingertip_distance_px"] for row in assigned) < 8.0,
        "all_states_are_hypotheses": bool(interactions)
        and all(row["epistemic_status"] == "hypothesis" for row in interactions),
        "grasp_then_hold_candidates": bool(assigned)
        and assigned[0]["event_candidate"] == "grasp_candidate"
        and all(row["interaction_state"] == "holding_candidate" for row in assigned[1:]),
        "session_complete": workspace.manifest.status == "complete",
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)

    original = cv2.imread(str(sample))
    capture = cv2.VideoCapture(str(workspace.review / "overlay.mp4"))
    ok, overlay = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Could not decode validation overlay")
    original = cv2.resize(original, (overlay.shape[1], overlay.shape[0]))
    comparison = cv2.hconcat([original, overlay])
    for x, label in (
        (0, "ORIGINAL WIKIMEDIA IMAGE"),
        (overlay.shape[1], "MOBOCAPTURE INTERACTION OUTPUT"),
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
        "module": "hand_object_interactions",
        "sample_url": SAMPLE_URL,
        "sample_page": SAMPLE_PAGE,
        "sample_license": "CC BY-SA 4.0; author PattayaPatrol",
        "sample_sha256": SAMPLE_SHA256,
        "assertions": assertions,
        "hands": len(hands),
        "regions": len(regions),
        "interaction_pairs": len(interactions),
        "assigned_pairs": len(assigned),
        "assigned_states": [row["interaction_state"] for row in assigned],
        "assigned_scores": [row["contact_likelihood"] for row in assigned],
        "comparison": str(comparison_path),
        "session": str(workspace.root),
    }
    (output_root / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("validation-output/interactions")
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
