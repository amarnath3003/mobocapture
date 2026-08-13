from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from mobocapture.assets import (
    GROUNDING_DINO_MODEL_ID,
    GROUNDING_DINO_REVISION,
    ensure_yunet_face_model,
)
from mobocapture.io import write_json
from mobocapture.models import EpistemicStatus, ProvenanceClass
from mobocapture.processors.base import Processor, ProcessorResult
from mobocapture.processors.objects import (
    PRIVACY_GROUNDING_CONCEPTS,
    _ObjectTrackAssigner,
    _canonical_label,
    _frame_batches,
    _grounding_dino_runtime,
    _inference_batch_size,
    _non_maximum_suppression,
    _release_grounding_dino_runtime,
)
from mobocapture.schemas import PRIVACY_FRAMES_SCHEMA, PRIVACY_REGIONS_SCHEMA
from mobocapture.session import SessionWorkspace


PRIVACY_CONCEPTS = PRIVACY_GROUNDING_CONCEPTS


def _expanded_box(box: list[float], width: int, height: int, fraction: float = 0.12):
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * fraction
    pad_y = (y2 - y1) * fraction
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x + 1)),
        min(height, int(y2 + pad_y + 1)),
    )


class PrivacyScannerProcessor(Processor):
    processor_id = "privacy_scanner"

    def process(self, workspace: SessionWorkspace) -> ProcessorResult:
        model_path, face_hash, _ = ensure_yunet_face_model()
        shared_grounding = workspace.runtime.pop("privacy_grounding_by_frame", None)
        runtime = _grounding_dino_runtime(workspace)
        torch = runtime.torch
        processor = runtime.processor
        model = runtime.model
        device = runtime.device
        batch_size = _inference_batch_size(workspace, device)
        face_detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), 0.55, 0.3, 5000
        )
        frame_table = pq.read_table(workspace.derived / "frame_index.parquet")
        timestamps = frame_table.column("timestamp_ns").to_pylist()
        capture = cv2.VideoCapture(str(workspace.raw_video))
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the copied video for privacy scanning")
        assigner = _ObjectTrackAssigner(maximum_age_frames=10, minimum_iou=0.12)
        rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        frame_index = 0
        try:
            for batch in _frame_batches(capture, batch_size):
                images = []
                face_candidates = []
                for _, frame in batch:
                    height, width = frame.shape[:2]
                    face_detector.setInputSize((width, height))
                    _, faces = face_detector.detect(frame)
                    frame_faces = []
                    if faces is not None:
                        for face in faces:
                            x, y, box_width, box_height = [
                                float(value) for value in face[:4]
                            ]
                            frame_faces.append(
                                {
                                    "label": "face",
                                    "confidence": float(face[14]),
                                    "bbox_xyxy_px": [
                                        x,
                                        y,
                                        x + box_width,
                                        y + box_height,
                                    ],
                                    "detector": "OpenCV YuNet",
                                    "model_revision": face_hash,
                                }
                            )
                    face_candidates.append(frame_faces)
                    if shared_grounding is None:
                        images.append(
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        )
                results = None
                if shared_grounding is None:
                    inputs = processor(
                        images=images,
                        text=[list(PRIVACY_CONCEPTS) for _ in images],
                        return_tensors="pt",
                    ).to(device)
                    with torch.inference_mode():
                        outputs = model(**inputs)
                    results = processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        threshold=0.30,
                        text_threshold=0.24,
                        target_sizes=[(image.height, image.width) for image in images],
                    )
                    del outputs
                result_items = results if results is not None else [None] * len(batch)
                for (current_index, frame), candidates, result in zip(
                    batch, face_candidates, result_items
                ):
                    height, width = frame.shape[:2]
                    timestamp_ns = (
                        timestamps[current_index] if current_index < len(timestamps) else 0
                    )
                    if shared_grounding is not None:
                        grounded = shared_grounding.get(current_index, [])
                    else:
                        labels = result.get("text_labels")
                        if labels is None:
                            labels = [
                                str(value) for value in result["labels"].tolist()
                            ]
                        scores = result["scores"].detach().cpu().tolist()
                        boxes = result["boxes"].detach().cpu().tolist()
                        grounded = [
                            {
                                "label": _canonical_label(
                                    str(label), list(PRIVACY_CONCEPTS)
                                ),
                                "confidence": float(score),
                                "bbox_xyxy_px": [float(value) for value in box],
                                "detector": "Grounding DINO",
                                "model_revision": GROUNDING_DINO_REVISION,
                            }
                            for label, score, box in zip(labels, scores, boxes)
                        ]
                        grounded = _non_maximum_suppression(grounded, 20)
                    candidates.extend(grounded)
                    track_ids = assigner.assign(current_index, candidates)
                    for region_index, (candidate, track_id) in enumerate(
                        zip(candidates, track_ids)
                    ):
                        rows.append(
                            {
                                "frame_index": current_index,
                                "timestamp_ns": timestamp_ns,
                                "region_index": region_index,
                                "track_id": f"privacy-{track_id.removeprefix('object-')}",
                                "category": candidate["label"],
                                "confidence": candidate["confidence"],
                                "bbox_xyxy_px": candidate["bbox_xyxy_px"],
                                "redaction_required": True,
                                "review_required": True,
                                "provenance": ProvenanceClass.OFFLINE_ESTIMATED.value,
                                "epistemic_status": EpistemicStatus.ESTIMATE.value,
                                "processor_id": self.processor_id,
                                "detector": candidate["detector"],
                                "model_revision": candidate["model_revision"],
                            }
                        )
                    frame_rows.append(
                        {
                            "frame_index": current_index,
                            "timestamp_ns": timestamp_ns,
                            "decode_ok": True,
                            "face_count": sum(
                                row["label"] == "face" for row in candidates
                            ),
                            "other_privacy_count": sum(
                                row["label"] != "face" for row in candidates
                            ),
                            "redaction_region_count": len(candidates),
                            "processor_id": self.processor_id,
                        }
                    )
                    frame_index = current_index + 1
        finally:
            capture.release()
            _release_grounding_dino_runtime(workspace)

        regions_path = workspace.derived / "privacy_regions.parquet"
        frames_path = workspace.derived / "privacy_frames.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=PRIVACY_REGIONS_SCHEMA),
            regions_path,
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(frame_rows, schema=PRIVACY_FRAMES_SCHEMA),
            frames_path,
            compression="zstd",
        )
        return ProcessorResult(
            outputs=[regions_path, frames_path],
            metrics={
                "frames_processed": frame_index,
                "privacy_regions": len(rows),
                "faces": sum(row["category"] == "face" for row in rows),
                "categories": sorted({row["category"] for row in rows}),
                "review_policy": "all automatic detections require human review before release",
                "identity_inference": "not performed",
                "face_model_sha256": face_hash,
                "grounding_dino_revision": GROUNDING_DINO_REVISION,
                "device": device,
                "inference_batch_size": batch_size,
                "inference_precision": "float32",
                "shared_object_visual_backbone": shared_grounding is not None,
            },
        )


class PrivacyRedactorProcessor(Processor):
    processor_id = "privacy_redactor"

    def process(self, workspace: SessionWorkspace) -> ProcessorResult:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required but was not found on PATH")
        grouped: dict[int, list[dict]] = defaultdict(list)
        regions_path = workspace.derived / "privacy_regions.parquet"
        for row in pq.read_table(regions_path).to_pylist():
            if row["redaction_required"]:
                grouped[row["frame_index"]].append(row)
        capture = cv2.VideoCapture(str(workspace.raw_video))
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the copied video for redaction")
        ok, first = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError("Cannot decode the first video frame for redaction")
        height, width = first.shape[:2]
        fps = workspace.manifest.video.nominal_frame_rate or capture.get(cv2.CAP_PROP_FPS) or 30
        output = workspace.review / "redacted.mp4"
        command = [
            ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s:v", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "pipe:0",
            "-i", str(workspace.raw_video), "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(output),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.stdin is None:
            capture.release()
            raise RuntimeError("Failed to open FFmpeg redaction pipe")
        frame_index = 0
        frame = first
        redactions = 0
        try:
            while True:
                for region in grouped.get(frame_index, []):
                    x1, y1, x2, y2 = _expanded_box(
                        region["bbox_xyxy_px"], width, height
                    )
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        kernel = max(15, (min(crop.shape[:2]) // 5) | 1)
                        frame[y1:y2, x1:x2] = cv2.GaussianBlur(crop, (kernel, kernel), 0)
                        redactions += 1
                process.stdin.write(frame.tobytes())
                frame_index += 1
                ok, frame = capture.read()
                if not ok:
                    break
        finally:
            capture.release()
            process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if process.wait() != 0:
            raise RuntimeError(f"FFmpeg redaction encoding failed: {stderr.strip()}")
        manifest_path = workspace.manifests / "redaction.json"
        write_json(
            manifest_path,
            {
                "schema_version": "0.1.0",
                "source": workspace.manifest.input.stored_path,
                "derivative": output.relative_to(workspace.root).as_posix(),
                "method": "expanded Gaussian blur over every automatic privacy candidate",
                "redaction_regions": redactions,
                "review_required": True,
                "warning": "Automatic redaction is not a release guarantee; human review is required.",
            },
        )
        return ProcessorResult(
            outputs=[output, manifest_path],
            metrics={
                "frames_rendered": frame_index,
                "redaction_regions_applied": redactions,
                "review_required": True,
                "original_preserved": True,
            },
        )
