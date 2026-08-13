from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from mobocapture.processors.base import Processor, ProcessorResult
from mobocapture.processors.objects import decode_binary_mask_rle
from mobocapture.session import SessionWorkspace


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
HAND_COLORS = {
    "left": (255, 185, 70),
    "right": (80, 175, 255),
    "unknown": (210, 210, 210),
}


def _draw_label(frame, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x - 4, y - height - 5), (x + width + 4, y + baseline + 3), (18, 18, 18), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _load_rows_by_frame(path: Path) -> dict[int, list[dict]]:
    if not path.is_file():
        return {}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in pq.read_table(path).to_pylist():
        grouped[row["frame_index"]].append(row)
    return dict(grouped)


def _object_color(track_id: str) -> tuple[int, int, int]:
    try:
        value = int(track_id.rsplit("-", 1)[-1])
    except ValueError:
        value = sum(track_id.encode("utf-8"))
    hue = (value * 47) % 180
    color = cv2.cvtColor(np.uint8([[[hue, 205, 245]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(channel) for channel in color)


def _load_flow_by_target(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    return {
        row["target_frame_index"]: row
        for row in pq.read_table(path).to_pylist()
    }


def _draw_dense_flow(frame, flow: np.ndarray) -> None:
    height, width = frame.shape[:2]
    if flow.shape[:2] != (height, width):
        return
    step = max(24, min(height, width) // 14)
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            dx, dy = (float(value) for value in flow[y, x])
            magnitude = (dx * dx + dy * dy) ** 0.5
            if not np.isfinite(magnitude) or magnitude < 0.45:
                continue
            scale = min(3.0, 18.0 / magnitude)
            end = (
                int(round(np.clip(x + dx * scale, 0, width - 1))),
                int(round(np.clip(y + dy * scale, 0, height - 1))),
            )
            cv2.arrowedLine(frame, (x, y), end, (255, 210, 72), 1, cv2.LINE_AA, tipLength=0.3)


def _draw_points(frame, points: list[dict]) -> None:
    for point in points:
        x, y = (int(round(value)) for value in point["xy_px"])
        previous = point.get("previous_xy_px")
        if previous is not None:
            old = tuple(int(round(value)) for value in previous)
            cv2.line(frame, old, (x, y), (98, 255, 143), 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 2, (35, 35, 35), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 1, (98, 255, 143), -1, cv2.LINE_AA)


def _draw_interactions(
    frame,
    interactions: list[dict],
    hands: list[dict],
    regions: list[dict],
) -> None:
    hand_lookup = {row["track_id"]: row for row in hands}
    region_lookup = {row["track_id"]: row for row in regions}
    for interaction in interactions:
        if not interaction["assigned_to_hand"]:
            continue
        hand = hand_lookup.get(interaction["hand_track_id"])
        region = region_lookup.get(interaction["object_track_id"])
        if hand is None or region is None or region["centroid_xy_px"] is None:
            continue
        wrist = tuple(int(round(value)) for value in hand["landmarks_2d_px"][0])
        centroid = tuple(int(round(value)) for value in region["centroid_xy_px"])
        color = (236, 92, 255)
        cv2.line(frame, wrist, centroid, color, 2, cv2.LINE_AA)
        midpoint = ((wrist[0] + centroid[0]) // 2, (wrist[1] + centroid[1]) // 2)
        _draw_label(
            frame,
            (
                f"{interaction['interaction_state']} "
                f"{interaction['contact_likelihood']:.2f} hypothesis"
            ),
            midpoint,
            color,
        )


def _draw_hand(frame, hand: dict) -> None:
    color = HAND_COLORS.get(hand["side"], HAND_COLORS["unknown"])
    points = [(int(round(x)), int(round(y))) for x, y in hand["landmarks_2d_px"]]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], color, 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        radius = 5 if index in {0, 4, 8, 12, 16, 20} else 3
        cv2.circle(frame, point, radius, (18, 18, 18), -1, cv2.LINE_AA)
        cv2.circle(frame, point, max(1, radius - 2), color, -1, cv2.LINE_AA)
    x1, y1, x2, y2 = [int(round(value)) for value in hand["bbox_xyxy_px"]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    confidence = hand.get("side_confidence", 0.0)
    _draw_label(
        frame,
        f"{hand['track_id']} {hand['side']} {confidence:.2f}",
        (max(6, x1), max(18, y1 - 5)),
        color,
    )


def _draw_region(frame, region: dict) -> None:
    color = _object_color(region["track_id"])
    mask = decode_binary_mask_rle(
        region["mask_rle_counts"], region["mask_height"], region["mask_width"]
    )
    if mask.shape == frame.shape[:2] and mask.any():
        tint = np.zeros_like(frame)
        tint[mask] = color
        frame[mask] = cv2.addWeighted(frame, 0.62, tint, 0.38, 0)[mask]
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, contours, -1, color, 2, cv2.LINE_AA)
    x1, y1, x2, y2 = [int(round(value)) for value in region["bbox_xyxy_px"]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    _draw_label(
        frame,
        f"{region['track_id']} {region['label']} {region['detection_confidence']:.2f}",
        (max(6, x1), max(18, y1 - 5)),
        color,
    )


class OverlayRendererProcessor(Processor):
    processor_id = "overlay_renderer"

    def process(self, workspace: SessionWorkspace) -> ProcessorResult:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required but was not found on PATH")
        if workspace.manifest.video is None:
            raise RuntimeError("Video metadata is missing")

        quality = pq.read_table(workspace.derived / "quality.parquet").to_pylist()
        hands_by_frame = _load_rows_by_frame(workspace.derived / "hands.parquet")
        regions_by_frame = _load_rows_by_frame(workspace.derived / "regions.parquet")
        points_by_frame = _load_rows_by_frame(workspace.derived / "point_tracks.parquet")
        flow_by_target = _load_flow_by_target(workspace.derived / "optical_flow.parquet")
        interactions_by_frame = _load_rows_by_frame(workspace.derived / "interactions.parquet")
        privacy_by_frame = _load_rows_by_frame(workspace.derived / "privacy_regions.parquet")
        capture = cv2.VideoCapture(str(workspace.raw_video))
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the copied input video")
        ok, first_frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError("Cannot decode the first video frame for overlay rendering")
        height, width = first_frame.shape[:2]
        fps = workspace.manifest.video.nominal_frame_rate or capture.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0:
            fps = 30.0

        output = workspace.review / "overlay.mp4"
        command = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "pipe:0",
            "-i",
            str(workspace.raw_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(output),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.stdin is None:
            capture.release()
            raise RuntimeError("Failed to open the FFmpeg overlay input pipe")

        frame_index = 0
        frame = first_frame
        try:
            while True:
                record = quality[frame_index] if frame_index < len(quality) else None
                timestamp_seconds = (
                    record["timestamp_ns"] / 1_000_000_000 if record is not None else frame_index / fps
                )
                _draw_label(
                    frame,
                    f"MoboCapture v0.1 | frame {frame_index} | {timestamp_seconds:.3f}s",
                    (12, 24),
                    (245, 245, 245),
                )
                if record is not None:
                    brightness = record.get("brightness_mean")
                    blur = record.get("blur_likelihood")
                    motion = record.get("camera_motion_category") or "n/a"
                    quality_text = (
                        f"brightness {brightness:.2f} | blur-risk {blur:.2f} | motion {motion}"
                        if brightness is not None and blur is not None
                        else "quality unavailable"
                    )
                    color = (80, 220, 120) if record.get("decode_ok") else (80, 80, 240)
                    _draw_label(frame, quality_text, (12, 48), color)
                hands = hands_by_frame.get(frame_index, [])
                regions = regions_by_frame.get(frame_index, [])
                points = points_by_frame.get(frame_index, [])
                flow_record = flow_by_target.get(frame_index)
                if flow_record is not None:
                    flow_file = workspace.root / flow_record["flow_path"]
                    with np.load(flow_file) as flow_data:
                        _draw_dense_flow(frame, flow_data["flow_xy_px"])
                _draw_points(frame, points)
                for region in regions:
                    _draw_region(frame, region)
                for hand in hands:
                    _draw_hand(frame, hand)
                interactions = interactions_by_frame.get(frame_index, [])
                _draw_interactions(frame, interactions, hands, regions)
                privacy_regions = privacy_by_frame.get(frame_index, [])
                for privacy in privacy_regions:
                    x1, y1, x2, y2 = [int(round(value)) for value in privacy["bbox_xyxy_px"]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (45, 45, 255), 2, cv2.LINE_AA)
                    _draw_label(
                        frame,
                        f"PRIVACY {privacy['category']} {privacy['confidence']:.2f}",
                        (max(6, x1), max(18, y1 - 5)),
                        (45, 45, 255),
                    )
                if hands_by_frame:
                    _draw_label(
                        frame,
                        f"hands {len(hands)} | observed landmarks only",
                        (12, 72),
                        (216, 255, 79),
                    )
                if regions_by_frame:
                    _draw_label(
                        frame,
                        f"objects {len(regions)} | boxes + observed masks",
                        (12, 96 if hands_by_frame else 72),
                        (94, 223, 255),
                    )
                if points_by_frame or flow_by_target:
                    label_y = 120 if hands_by_frame and regions_by_frame else (
                        96 if hands_by_frame or regions_by_frame else 72
                    )
                    mean_flow = (
                        flow_record["mean_magnitude_px"] if flow_record is not None else 0.0
                    )
                    _draw_label(
                        frame,
                        f"motion points {len(points)} | mean flow {mean_flow:.2f}px/frame",
                        (12, label_y),
                        (98, 255, 143),
                    )
                if interactions_by_frame:
                    assigned = sum(row["assigned_to_hand"] for row in interactions)
                    stack_count = sum(
                        bool(value)
                        for value in (
                            hands_by_frame,
                            regions_by_frame,
                            points_by_frame or flow_by_target,
                        )
                    )
                    _draw_label(
                        frame,
                        f"interaction hypotheses {assigned}/{len(interactions)} assigned",
                        (12, 72 + 24 * stack_count),
                        (236, 92, 255),
                    )
                if privacy_by_frame:
                    _draw_label(
                        frame,
                        f"privacy candidates {len(privacy_regions)} | review required",
                        (12, height - 18),
                        (45, 45, 255),
                    )
                process.stdin.write(frame.tobytes())
                frame_index += 1
                ok, frame = capture.read()
                if not ok:
                    break
        except BrokenPipeError as error:
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            raise RuntimeError(f"FFmpeg overlay encoding failed: {stderr.strip()}") from error
        finally:
            capture.release()
            process.stdin.close()

        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg overlay encoding failed: {stderr.strip()}")
        return ProcessorResult(
            outputs=[output],
            metrics={
                "rendered_frames": frame_index,
                "visualization_frame_rate": fps,
                "timing_note": "Review video is CFR; canonical timestamps remain in frame_index.parquet",
                "encoder": "ffmpeg/libx264",
                "hand_overlay_enabled": bool(hands_by_frame),
                "hand_observations_rendered": sum(len(rows) for rows in hands_by_frame.values()),
                "object_overlay_enabled": bool(regions_by_frame),
                "object_observations_rendered": sum(
                    len(rows) for rows in regions_by_frame.values()
                ),
                "point_overlay_enabled": bool(points_by_frame),
                "point_observations_rendered": sum(
                    len(rows) for rows in points_by_frame.values()
                ),
                "dense_flow_overlay_enabled": bool(flow_by_target),
                "dense_flow_pairs_rendered": len(flow_by_target),
                "interaction_overlay_enabled": bool(interactions_by_frame),
                "interaction_candidates_rendered": sum(
                    len(rows) for rows in interactions_by_frame.values()
                ),
                "privacy_overlay_enabled": bool(privacy_by_frame),
                "privacy_candidates_rendered": sum(
                    len(rows) for rows in privacy_by_frame.values()
                ),
            },
        )
