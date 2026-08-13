from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mobocapture.models import EpistemicStatus, ProvenanceClass
from mobocapture.processors.base import Processor, ProcessorResult
from mobocapture.processors.objects import decode_binary_mask_rle
from mobocapture.schemas import INTERACTION_FRAMES_SCHEMA, INTERACTIONS_SCHEMA
from mobocapture.session import SessionWorkspace


METHOD = (
    "evidence heuristic: fingertip-to-mask proximity + mask inclusion + "
    "hand/object overlap + frame-to-frame co-motion"
)
FINGERTIP_INDICES = (4, 8, 12, 16, 20)
PALM_INDICES = (0, 5, 9, 13, 17)


def _rows_by_frame(path) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    if path.is_file():
        for row in pq.read_table(path).to_pylist():
            grouped[row["frame_index"]].append(row)
    return dict(grouped)


def _overlap_fraction(hand_box: list[float], object_box: list[float]) -> float:
    x1 = max(hand_box[0], object_box[0])
    y1 = max(hand_box[1], object_box[1])
    x2 = min(hand_box[2], object_box[2])
    y2 = min(hand_box[3], object_box[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    hand_area = max(0.0, hand_box[2] - hand_box[0]) * max(
        0.0, hand_box[3] - hand_box[1]
    )
    return intersection / hand_area if hand_area > 0 else 0.0


def _motion_similarity(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None:
        return None
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 0.5 or second_norm < 0.5:
        return None
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))


def _palm_center(hand: dict) -> np.ndarray:
    points = np.asarray(hand["landmarks_2d_px"], dtype=np.float32)
    return points[list(PALM_INDICES)].mean(axis=0)


class InteractionInferenceProcessor(Processor):
    processor_id = "interaction_inference"

    def process(self, workspace: SessionWorkspace) -> ProcessorResult:
        hands_by_frame = _rows_by_frame(workspace.derived / "hands.parquet")
        regions_by_frame = _rows_by_frame(workspace.derived / "regions.parquet")
        frame_table = pq.read_table(workspace.derived / "frame_index.parquet")
        frame_records = frame_table.to_pylist()
        rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        previous_hand_centers: dict[str, np.ndarray] = {}
        previous_object_centers: dict[str, np.ndarray] = {}
        previous_distances: dict[tuple[str, str], float] = {}
        active_pairs: dict[tuple[str, str], bool] = {}

        for frame_record in frame_records:
            frame_index = frame_record["frame_index"]
            timestamp_ns = frame_record["timestamp_ns"]
            width = frame_record["width"]
            height = frame_record["height"]
            diagonal = math.hypot(width, height)
            hands = hands_by_frame.get(frame_index, [])
            regions = regions_by_frame.get(frame_index, [])
            current_hand_centers = {
                hand["track_id"]: _palm_center(hand) for hand in hands
            }
            current_object_centers = {
                region["track_id"]: np.asarray(region["centroid_xy_px"], dtype=np.float32)
                for region in regions
                if region["centroid_xy_px"] is not None
            }
            frame_candidates: list[dict[str, Any]] = []
            candidates_by_hand: dict[str, list[int]] = defaultdict(list)
            region_masks = {}
            for region in regions:
                mask = decode_binary_mask_rle(
                    region["mask_rle_counts"],
                    region["mask_height"],
                    region["mask_width"],
                )
                region_masks[region["track_id"]] = (
                    mask,
                    cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3),
                )

            for hand in hands:
                fingertips = np.asarray(hand["landmarks_2d_px"], dtype=np.float32)[
                    list(FINGERTIP_INDICES)
                ]
                hand_center = current_hand_centers[hand["track_id"]]
                previous_hand = previous_hand_centers.get(hand["track_id"])
                hand_motion = hand_center - previous_hand if previous_hand is not None else None
                for region in regions:
                    mask, outside_distance = region_masks[region["track_id"]]
                    distances = []
                    inside_count = 0
                    for x_value, y_value in fingertips:
                        x = int(np.clip(round(float(x_value)), 0, width - 1))
                        y = int(np.clip(round(float(y_value)), 0, height - 1))
                        inside_count += int(mask[y, x])
                        distances.append(float(outside_distance[y, x]))
                    minimum_distance = min(distances)
                    normalized_distance = minimum_distance / diagonal if diagonal else 1.0
                    proximity = math.exp(-normalized_distance / 0.035)
                    overlap = _overlap_fraction(
                        hand["bbox_xyxy_px"], region["bbox_xyxy_px"]
                    )
                    object_center = current_object_centers.get(region["track_id"])
                    previous_object = previous_object_centers.get(region["track_id"])
                    object_motion = (
                        object_center - previous_object
                        if object_center is not None and previous_object is not None
                        else None
                    )
                    motion_similarity = _motion_similarity(hand_motion, object_motion)
                    contact_likelihood = float(
                        np.clip(
                            0.45 * proximity
                            + 0.30 * (inside_count / len(FINGERTIP_INDICES))
                            + 0.15 * overlap
                            + 0.10 * (motion_similarity or 0.0),
                            0.0,
                            1.0,
                        )
                    )
                    assignment_confidence = contact_likelihood * math.sqrt(
                        max(0.0, float(region["detection_confidence"]))
                    )
                    pair = (hand["track_id"], region["track_id"])
                    previous_distance = previous_distances.get(pair)
                    approaching = (
                        previous_distance is not None
                        and previous_distance - normalized_distance > 0.003
                    )
                    candidate_index = len(frame_candidates)
                    candidates_by_hand[hand["track_id"]].append(candidate_index)
                    frame_candidates.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_ns": timestamp_ns,
                            "hand_track_id": hand["track_id"],
                            "hand_side": hand["side"],
                            "object_track_id": region["track_id"],
                            "object_label": region["label"],
                            "minimum_fingertip_distance_px": minimum_distance,
                            "normalized_fingertip_distance": normalized_distance,
                            "fingertips_inside_mask": inside_count,
                            "hand_bbox_overlap_fraction": overlap,
                            "motion_similarity": motion_similarity,
                            "contact_likelihood": contact_likelihood,
                            "assignment_confidence": assignment_confidence,
                            "assigned_to_hand": False,
                            "interaction_state": "approaching" if approaching else "separate",
                            "event_candidate": None,
                            "provenance": ProvenanceClass.OFFLINE_ESTIMATED.value,
                            "epistemic_status": EpistemicStatus.HYPOTHESIS.value,
                            "processor_id": self.processor_id,
                            "method": METHOD,
                        }
                    )
                    previous_distances[pair] = normalized_distance

            assigned_count = 0
            for hand_id, indices in candidates_by_hand.items():
                best_index = max(
                    indices,
                    key=lambda index: frame_candidates[index]["assignment_confidence"],
                )
                candidate = frame_candidates[best_index]
                if candidate["assignment_confidence"] < 0.20:
                    continue
                candidate["assigned_to_hand"] = True
                assigned_count += 1
                pair = (hand_id, candidate["object_track_id"])
                was_active = active_pairs.get(pair, False)
                likelihood = candidate["contact_likelihood"]
                if likelihood >= 0.50 and not was_active:
                    candidate["interaction_state"] = "contact_candidate"
                    candidate["event_candidate"] = "grasp_candidate"
                    active_pairs[pair] = True
                elif was_active and likelihood >= 0.30:
                    candidate["interaction_state"] = "holding_candidate"
                elif was_active:
                    candidate["interaction_state"] = "release_candidate"
                    candidate["event_candidate"] = "release_candidate"
                    active_pairs[pair] = False
                elif likelihood >= 0.38:
                    candidate["interaction_state"] = "contact_candidate"

            rows.extend(frame_candidates)
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ns": timestamp_ns,
                    "hand_count": len(hands),
                    "object_count": len(regions),
                    "candidate_pair_count": len(frame_candidates),
                    "assigned_pair_count": assigned_count,
                    "processor_id": self.processor_id,
                }
            )
            previous_hand_centers = current_hand_centers
            previous_object_centers = current_object_centers

        interactions_path = workspace.derived / "interactions.parquet"
        frames_path = workspace.derived / "interaction_frames.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=INTERACTIONS_SCHEMA),
            interactions_path,
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(frame_rows, schema=INTERACTION_FRAMES_SCHEMA),
            frames_path,
            compression="zstd",
        )
        return ProcessorResult(
            outputs=[interactions_path, frames_path],
            metrics={
                "frames_processed": len(frame_rows),
                "candidate_pairs": len(rows),
                "assigned_pairs": sum(row["assigned_to_hand"] for row in rows),
                "grasp_candidates": sum(
                    row["event_candidate"] == "grasp_candidate" for row in rows
                ),
                "release_candidates": sum(
                    row["event_candidate"] == "release_candidate" for row in rows
                ),
                "method": METHOD,
                "truth_note": "All interaction states are hypotheses derived from RGB evidence.",
            },
        )
