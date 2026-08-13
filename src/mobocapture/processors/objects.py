from __future__ import annotations

import gc
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from mobocapture.assets import (
    GROUNDING_DINO_MODEL_ID,
    GROUNDING_DINO_REVISION,
    OMDET_TURBO_MODEL_ID,
    OMDET_TURBO_REVISION,
    SAM2_MODEL_ID,
    SAM2_REVISION,
    huggingface_cache_root,
)
from mobocapture.models import EpistemicStatus, ProvenanceClass
from mobocapture.processors.base import Processor, ProcessorResult
from mobocapture.schemas import (
    OBJECT_DETECTION_FRAMES_SCHEMA,
    OBJECT_DETECTIONS_SCHEMA,
    OBJECT_FRAMES_SCHEMA,
    REGIONS_SCHEMA,
)
from mobocapture.session import SessionWorkspace


DEFAULT_OBJECT_CONCEPTS = (
    "person",
    "cup",
    "bottle",
    "bowl",
    "plate",
    "box",
    "bag",
    "book",
    "phone",
    "laptop",
    "chair",
    "table",
    "tool",
    "knife",
    "spoon",
    "fork",
    "scissors",
    "cloth",
    "drawer",
    "door",
    "vehicle",
    "truck",
    "wheel",
)

PRIVACY_GROUNDING_CONCEPTS = (
    "license plate",
    "computer screen",
    "phone screen",
    "document",
    "paper",
    "ID card",
    "badge",
    "mirror",
)


def _transformers_classes():
    # Some environments contain an unrelated, incompatible torchao install.
    # These models are not quantized, so prevent Transformers from importing that
    # optional integration while loading the standard PyTorch classes.
    import transformers.utils.import_utils as import_utils

    import_utils._torchao_available = False
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
        Sam2Processor,
    )

    return AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Processor, Sam2Model


def _torch_device(torch) -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class _GroundingDinoRuntime:
    torch: Any
    processor: Any
    model: Any
    device: str


@dataclass
class _OmDetRuntime:
    torch: Any
    processor: Any
    model: Any
    device: str


def _detector_backend(workspace: SessionWorkspace) -> str:
    backend = str(
        workspace.options.get("objects", {}).get("detector_backend", "omdet_turbo")
    ).strip().lower()
    if backend not in {"omdet_turbo", "grounding_dino"}:
        raise ValueError(
            "objects.detector_backend must be 'omdet_turbo' or 'grounding_dino'"
        )
    return backend


def _omdet_runtime(workspace: SessionWorkspace) -> _OmDetRuntime:
    key = "omdet_turbo_runtime"
    cached = workspace.runtime.get(key)
    if cached is not None:
        return cached
    import torch
    import transformers.utils.import_utils as import_utils

    import_utils._torchao_available = False
    from transformers import AutoProcessor, OmDetTurboForObjectDetection

    cache = huggingface_cache_root()
    cache.mkdir(parents=True, exist_ok=True)
    device = _torch_device(torch)
    processor = AutoProcessor.from_pretrained(
        OMDET_TURBO_MODEL_ID,
        revision=OMDET_TURBO_REVISION,
        cache_dir=cache,
        use_fast=False,
    )
    model = OmDetTurboForObjectDetection.from_pretrained(
        OMDET_TURBO_MODEL_ID,
        revision=OMDET_TURBO_REVISION,
        cache_dir=cache,
    )
    # Transformers 4.57 can leave Swin's deterministic attention-mask buffers
    # on the meta device. They are regenerated during forward; materializing
    # them makes the pinned checkpoint movable on current stable PyTorch.
    for module in model.modules():
        mask = getattr(module, "attn_mask", None)
        if mask is not None and mask.device.type == "meta":
            module.attn_mask = torch.empty_like(mask, device="cpu")
    model = model.to(device).eval()
    runtime = _OmDetRuntime(torch, processor, model, device)
    workspace.runtime[key] = runtime
    return runtime


def _release_omdet_runtime(workspace: SessionWorkspace) -> None:
    runtime = workspace.runtime.pop("omdet_turbo_runtime", None)
    if runtime is None:
        return
    del runtime.model
    if runtime.device == "cuda":
        runtime.torch.cuda.empty_cache()
    gc.collect()


def _grounding_dino_runtime(workspace: SessionWorkspace) -> _GroundingDinoRuntime:
    key = "grounding_dino_runtime"
    cached = workspace.runtime.get(key)
    if cached is not None:
        return cached
    import torch

    AutoProcessor, AutoModel, _, _ = _transformers_classes()
    cache = huggingface_cache_root()
    cache.mkdir(parents=True, exist_ok=True)
    device = _torch_device(torch)
    model = AutoModel.from_pretrained(
        GROUNDING_DINO_MODEL_ID,
        revision=GROUNDING_DINO_REVISION,
        cache_dir=cache,
    ).to(device).eval()
    # Prompts do not change between frames. Grounding DINO otherwise runs its
    # BERT text backbone for the same token IDs on every batch. In eval mode
    # those outputs are deterministic, so retaining them avoids exact duplicate
    # work without changing any tensor values supplied to the fusion encoder.
    text_backbone = model.model.text_backbone
    original_text_forward = text_backbone.forward
    text_output_cache = {}

    def cached_text_forward(input_ids, *args, **kwargs):
        key = (tuple(input_ids.shape), tuple(input_ids[0].tolist()))
        value = text_output_cache.get(key)
        if value is None:
            value = original_text_forward(input_ids, *args, **kwargs)
            text_output_cache[key] = value
        return value

    text_backbone.forward = cached_text_forward
    runtime = _GroundingDinoRuntime(
        torch=torch,
        processor=AutoProcessor.from_pretrained(
            GROUNDING_DINO_MODEL_ID,
            revision=GROUNDING_DINO_REVISION,
            cache_dir=cache,
        ),
        model=model,
        device=device,
    )
    workspace.runtime[key] = runtime
    return runtime


def _release_grounding_dino_runtime(workspace: SessionWorkspace) -> None:
    runtime = workspace.runtime.pop("grounding_dino_runtime", None)
    if runtime is None:
        return
    del runtime.model
    if runtime.device == "cuda":
        runtime.torch.cuda.empty_cache()
    gc.collect()


def _inference_batch_size(workspace: SessionWorkspace, device: str) -> int:
    configured = workspace.options.get("performance", {}).get("gpu_batch_size")
    if configured is not None:
        return max(1, min(int(configured), 8))
    # Two full-resolution samples fit comfortably on an 8 GB RTX 3070. Larger
    # batches used more memory and were slower for Grounding DINO in profiling.
    return 2 if device == "cuda" else 1


def _omdet_batch_size(workspace: SessionWorkspace, device: str) -> int:
    configured = workspace.options.get("performance", {}).get("gpu_batch_size")
    if configured is not None:
        return max(1, min(int(configured), 8))
    return 4 if device == "cuda" else 1


def _frame_batches(capture: cv2.VideoCapture, batch_size: int):
    frame_index = 0
    while True:
        batch = []
        for _ in range(batch_size):
            decode_ok, frame = capture.read()
            if not decode_ok:
                break
            batch.append((frame_index, frame))
            frame_index += 1
        if batch:
            yield batch
        if len(batch) < batch_size:
            break


def _object_options(workspace: SessionWorkspace) -> dict[str, Any]:
    configured = workspace.options.get("objects", {})
    raw_concepts = configured.get("concepts") or DEFAULT_OBJECT_CONCEPTS
    concepts = []
    for value in raw_concepts:
        concept = " ".join(str(value).strip().lower().split())
        if concept and concept not in concepts:
            concepts.append(concept)
    if not concepts:
        raise ValueError("Object detection needs at least one non-empty concept")
    if len(concepts) > 40 or any(len(item) > 80 for item in concepts):
        raise ValueError("Use at most 40 object concepts, each no longer than 80 characters")
    return {
        "concepts": concepts,
        "box_threshold": float(configured.get("box_threshold", 0.35)),
        "text_threshold": float(configured.get("text_threshold", 0.25)),
        "max_objects_per_frame": int(configured.get("max_objects_per_frame", 16)),
    }


def _box_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _non_maximum_suppression(detections: list[dict], maximum: int) -> list[dict]:
    kept: list[dict] = []
    for detection in sorted(detections, key=lambda row: row["confidence"], reverse=True):
        duplicate = any(
            prior["label"] == detection["label"]
            and _box_iou(prior["bbox_xyxy_px"], detection["bbox_xyxy_px"]) >= 0.7
            for prior in kept
        )
        if not duplicate:
            kept.append(detection)
        if len(kept) >= maximum:
            break
    return kept


def _canonical_label(raw_label: str, concepts: list[str]) -> str:
    label = " ".join(raw_label.strip().lower().split())
    matches = [concept for concept in concepts if concept in label]
    return max(matches, key=len) if matches else label


@dataclass
class _ObjectTrack:
    track_id: str
    label: str
    bbox: list[float]
    last_frame: int


class _ObjectTrackAssigner:
    def __init__(self, maximum_age_frames: int = 8, minimum_iou: float = 0.15):
        self.maximum_age_frames = maximum_age_frames
        self.minimum_iou = minimum_iou
        self._tracks: list[_ObjectTrack] = []
        self._next_id = 1

    def assign(self, frame_index: int, detections: list[dict]) -> list[str]:
        self._tracks = [
            track
            for track in self._tracks
            if frame_index - track.last_frame <= self.maximum_age_frames
        ]
        assignments: list[str] = []
        used: set[str] = set()
        for detection in detections:
            candidates = [
                (_box_iou(track.bbox, detection["bbox_xyxy_px"]), track)
                for track in self._tracks
                if track.track_id not in used and track.label == detection["label"]
            ]
            candidates = [item for item in candidates if item[0] >= self.minimum_iou]
            if candidates:
                _, track = max(candidates, key=lambda item: item[0])
            else:
                track = _ObjectTrack(
                    track_id=f"object-{self._next_id:04d}",
                    label=detection["label"],
                    bbox=detection["bbox_xyxy_px"],
                    last_frame=frame_index,
                )
                self._next_id += 1
                self._tracks.append(track)
            track.bbox = detection["bbox_xyxy_px"]
            track.last_frame = frame_index
            used.add(track.track_id)
            assignments.append(track.track_id)
        return assignments


def encode_binary_mask_rle(mask: np.ndarray) -> list[int]:
    """Encode a mask as uncompressed COCO-style column-major run lengths."""

    pixels = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    if pixels.size == 0:
        return [0]
    changes = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    counts = np.diff(
        np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                changes,
                np.asarray([pixels.size], dtype=np.int64),
            )
        )
    )
    if pixels[0] != 0:
        counts = np.concatenate((np.asarray([0], dtype=np.int64), counts))
    return counts.tolist()


def decode_binary_mask_rle(counts: list[int], height: int, width: int) -> np.ndarray:
    run_lengths = np.asarray(counts, dtype=np.int64)
    if np.any(run_lengths < 0) or int(run_lengths.sum()) != height * width:
        raise ValueError("Mask RLE size does not match its declared dimensions")
    values = np.repeat(
        np.arange(run_lengths.size, dtype=np.uint8) & 1, run_lengths
    )
    return values.reshape((height, width), order="F").astype(bool)


class ObjectDetectorProcessor(Processor):
    processor_id = "object_detector"

    def process(self, workspace: SessionWorkspace) -> ProcessorResult:
        if _detector_backend(workspace) == "omdet_turbo":
            return self._process_omdet(workspace)
        return self._process_grounding_dino(workspace)

    def _process_omdet(self, workspace: SessionWorkspace) -> ProcessorResult:
        settings = _object_options(workspace)
        runtime = _omdet_runtime(workspace)
        torch = runtime.torch
        processor = runtime.processor
        model = runtime.model
        device = runtime.device
        batch_size = _omdet_batch_size(workspace, device)
        score_threshold = float(
            workspace.options.get("objects", {}).get("omdet_score_threshold", 0.20)
        )
        frame_table = pq.read_table(workspace.derived / "frame_index.parquet")
        timestamps = frame_table.column("timestamp_ns").to_pylist()
        capture = cv2.VideoCapture(str(workspace.raw_video))
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the copied video for object detection")
        rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        frame_index = 0
        try:
            for batch in _frame_batches(capture, batch_size):
                images = [
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    for _, frame in batch
                ]
                concepts_batch = [settings["concepts"] for _ in images]
                inputs = processor(
                    images=images, text=concepts_batch, return_tensors="pt"
                ).to(device)
                with torch.inference_mode():
                    outputs = model(**inputs)
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    text_labels=concepts_batch,
                    threshold=score_threshold,
                    nms_threshold=0.5,
                    target_sizes=[(image.height, image.width) for image in images],
                    max_num_det=settings["max_objects_per_frame"],
                )
                for (current_index, _), result in zip(batch, results):
                    timestamp_ns = (
                        timestamps[current_index] if current_index < len(timestamps) else 0
                    )
                    labels = result.get("text_labels", result.get("classes", []))
                    scores = result["scores"].detach().cpu().tolist()
                    boxes = result["boxes"].detach().cpu().tolist()
                    detections = [
                        {
                            "label": _canonical_label(str(label), settings["concepts"]),
                            "confidence": float(score),
                            "bbox_xyxy_px": [float(value) for value in box],
                        }
                        for label, score, box in zip(labels, scores, boxes)
                    ]
                    for detection_index, detection in enumerate(detections):
                        rows.append(
                            {
                                "frame_index": current_index,
                                "timestamp_ns": timestamp_ns,
                                "detection_index": detection_index,
                                **detection,
                                "prompt_concepts": settings["concepts"],
                                "provenance": ProvenanceClass.OFFLINE_ESTIMATED.value,
                                "epistemic_status": EpistemicStatus.ESTIMATE.value,
                                "processor_id": self.processor_id,
                                "model_id": OMDET_TURBO_MODEL_ID,
                                "model_revision": OMDET_TURBO_REVISION,
                            }
                        )
                    frame_rows.append(
                        {
                            "frame_index": current_index,
                            "timestamp_ns": timestamp_ns,
                            "decode_ok": True,
                            "detection_count": len(detections),
                            "processor_id": self.processor_id,
                            "model_id": OMDET_TURBO_MODEL_ID,
                            "model_revision": OMDET_TURBO_REVISION,
                        }
                    )
                    frame_index = current_index + 1
        finally:
            capture.release()
            _release_omdet_runtime(workspace)
        for missing_index in range(frame_index, len(timestamps)):
            frame_rows.append(
                {
                    "frame_index": missing_index,
                    "timestamp_ns": timestamps[missing_index],
                    "decode_ok": False,
                    "detection_count": 0,
                    "processor_id": self.processor_id,
                    "model_id": OMDET_TURBO_MODEL_ID,
                    "model_revision": OMDET_TURBO_REVISION,
                }
            )
        detections_path = workspace.derived / "object_detections.parquet"
        frames_path = workspace.derived / "object_detection_frames.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=OBJECT_DETECTIONS_SCHEMA),
            detections_path,
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(frame_rows, schema=OBJECT_DETECTION_FRAMES_SCHEMA),
            frames_path,
            compression="zstd",
        )
        return ProcessorResult(
            outputs=[detections_path, frames_path],
            metrics={
                "frames_processed": frame_index,
                "object_detections": len(rows),
                "frames_with_detections": sum(
                    row["detection_count"] > 0 for row in frame_rows
                ),
                "prompt_concepts": settings["concepts"],
                "score_threshold": score_threshold,
                "nms_threshold": 0.5,
                "model_id": OMDET_TURBO_MODEL_ID,
                "model_revision": OMDET_TURBO_REVISION,
                "detector_backend": "omdet_turbo",
                "device": device,
                "inference_batch_size": batch_size,
                "inference_precision": "float32",
            },
        )

    def _process_grounding_dino(self, workspace: SessionWorkspace) -> ProcessorResult:
        settings = _object_options(workspace)
        runtime = _grounding_dino_runtime(workspace)
        torch = runtime.torch
        processor = runtime.processor
        model = runtime.model
        device = runtime.device
        batch_size = _inference_batch_size(workspace, device)

        frame_table = pq.read_table(workspace.derived / "frame_index.parquet")
        timestamps = frame_table.column("timestamp_ns").to_pylist()
        capture = cv2.VideoCapture(str(workspace.raw_video))
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the copied video for object detection")
        rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        frame_index = 0
        succeeded = False
        planned = workspace.runtime.get("planned_processor_ids", frozenset())
        privacy_enabled = "privacy_scanner" in planned
        privacy_grounding_by_frame: dict[int, list[dict[str, Any]]] = {}
        try:
            for batch in _frame_batches(capture, batch_size):
                images = [
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    for _, frame in batch
                ]
                inputs = processor(
                    images=images,
                    text=[settings["concepts"] for _ in images],
                    return_tensors="pt",
                ).to(device)
                privacy_inputs = None
                backbone = None
                original_backbone_forward = None
                cached_features = []
                reuse_features = [False]
                if privacy_enabled:
                    privacy_inputs = processor(
                        images=images,
                        text=[list(PRIVACY_GROUNDING_CONCEPTS) for _ in images],
                        return_tensors="pt",
                    ).to(device)
                    # Image preprocessing is independent of the text prompt.
                    # Reusing these exact tensors also makes the cached visual
                    # backbone output valid for the privacy prompt.
                    privacy_inputs["pixel_values"] = inputs["pixel_values"]
                    privacy_inputs["pixel_mask"] = inputs["pixel_mask"]
                    backbone = model.model.backbone
                    original_backbone_forward = backbone.forward

                    def cached_backbone_forward(pixel_values, pixel_mask):
                        if reuse_features[0]:
                            return cached_features[0]
                        value = original_backbone_forward(pixel_values, pixel_mask)
                        cached_features.append(value)
                        return value

                    backbone.forward = cached_backbone_forward
                try:
                    with torch.inference_mode():
                        outputs = model(**inputs)
                    results = processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        threshold=settings["box_threshold"],
                        text_threshold=settings["text_threshold"],
                        target_sizes=[(image.height, image.width) for image in images],
                    )
                    del outputs
                except Exception:
                    if backbone is not None and original_backbone_forward is not None:
                        backbone.forward = original_backbone_forward
                    raise
                for (current_index, _), result in zip(batch, results):
                    timestamp_ns = (
                        timestamps[current_index] if current_index < len(timestamps) else 0
                    )
                    labels = result.get("text_labels")
                    if labels is None:
                        labels = [str(value) for value in result["labels"].tolist()]
                    scores = result["scores"].detach().cpu().tolist()
                    boxes = result["boxes"].detach().cpu().tolist()
                    candidates = [
                        {
                            "label": _canonical_label(str(label), settings["concepts"]),
                            "confidence": float(score),
                            "bbox_xyxy_px": [float(value) for value in box],
                        }
                        for label, score, box in zip(labels, scores, boxes)
                    ]
                    detections = _non_maximum_suppression(
                        candidates, settings["max_objects_per_frame"]
                    )
                    for detection_index, detection in enumerate(detections):
                        rows.append(
                            {
                                "frame_index": current_index,
                                "timestamp_ns": timestamp_ns,
                                "detection_index": detection_index,
                                **detection,
                                "prompt_concepts": settings["concepts"],
                                "provenance": ProvenanceClass.OFFLINE_ESTIMATED.value,
                                "epistemic_status": EpistemicStatus.ESTIMATE.value,
                                "processor_id": self.processor_id,
                                "model_id": GROUNDING_DINO_MODEL_ID,
                                "model_revision": GROUNDING_DINO_REVISION,
                            }
                        )
                    frame_rows.append(
                        {
                            "frame_index": current_index,
                            "timestamp_ns": timestamp_ns,
                            "decode_ok": True,
                            "detection_count": len(detections),
                            "processor_id": self.processor_id,
                            "model_id": GROUNDING_DINO_MODEL_ID,
                            "model_revision": GROUNDING_DINO_REVISION,
                        }
                    )
                    frame_index = current_index + 1
                del results
                if privacy_inputs is not None:
                    reuse_features[0] = True
                    try:
                        with torch.inference_mode():
                            privacy_outputs = model(**privacy_inputs)
                        privacy_results = processor.post_process_grounded_object_detection(
                            privacy_outputs,
                            privacy_inputs.input_ids,
                            threshold=0.30,
                            text_threshold=0.24,
                            target_sizes=[
                                (image.height, image.width) for image in images
                            ],
                        )
                        del privacy_outputs
                        for (current_index, _), result in zip(batch, privacy_results):
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
                                        str(label), list(PRIVACY_GROUNDING_CONCEPTS)
                                    ),
                                    "confidence": float(score),
                                    "bbox_xyxy_px": [float(value) for value in box],
                                    "detector": "Grounding DINO",
                                    "model_revision": GROUNDING_DINO_REVISION,
                                }
                                for label, score, box in zip(labels, scores, boxes)
                            ]
                            privacy_grounding_by_frame[current_index] = (
                                _non_maximum_suppression(grounded, 20)
                            )
                    finally:
                        backbone.forward = original_backbone_forward
            succeeded = True
            if privacy_enabled:
                workspace.runtime["privacy_grounding_by_frame"] = (
                    privacy_grounding_by_frame
                )
        finally:
            capture.release()
            if not succeeded or "privacy_scanner" not in planned:
                _release_grounding_dino_runtime(workspace)

        for missing_index in range(frame_index, len(timestamps)):
            frame_rows.append(
                {
                    "frame_index": missing_index,
                    "timestamp_ns": timestamps[missing_index],
                    "decode_ok": False,
                    "detection_count": 0,
                    "processor_id": self.processor_id,
                    "model_id": GROUNDING_DINO_MODEL_ID,
                    "model_revision": GROUNDING_DINO_REVISION,
                }
            )

        detections_path = workspace.derived / "object_detections.parquet"
        frames_path = workspace.derived / "object_detection_frames.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=OBJECT_DETECTIONS_SCHEMA),
            detections_path,
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(frame_rows, schema=OBJECT_DETECTION_FRAMES_SCHEMA),
            frames_path,
            compression="zstd",
        )
        return ProcessorResult(
            outputs=[detections_path, frames_path],
            metrics={
                "frames_processed": frame_index,
                "object_detections": len(rows),
                "frames_with_detections": sum(row["detection_count"] > 0 for row in frame_rows),
                "prompt_concepts": settings["concepts"],
                "box_threshold": settings["box_threshold"],
                "text_threshold": settings["text_threshold"],
                "model_id": GROUNDING_DINO_MODEL_ID,
                "model_revision": GROUNDING_DINO_REVISION,
                "device": device,
                "inference_batch_size": batch_size,
                "inference_precision": "float32",
                "shared_privacy_visual_backbone": privacy_enabled,
                "cached_text_encoder_outputs": True,
            },
        )


class ObjectSegmenterProcessor(Processor):
    processor_id = "object_segmenter"

    def process(self, workspace: SessionWorkspace) -> ProcessorResult:
        import torch

        _, _, Sam2Processor, Sam2Model = _transformers_classes()
        cache = huggingface_cache_root()
        cache.mkdir(parents=True, exist_ok=True)
        device = _torch_device(torch)
        processor = Sam2Processor.from_pretrained(
            SAM2_MODEL_ID,
            revision=SAM2_REVISION,
            cache_dir=cache,
        )
        model = Sam2Model.from_pretrained(
            SAM2_MODEL_ID,
            revision=SAM2_REVISION,
            cache_dir=cache,
        ).to(device).eval()

        detections_by_frame: dict[int, list[dict]] = defaultdict(list)
        detections_path = workspace.derived / "object_detections.parquet"
        for row in pq.read_table(detections_path).to_pylist():
            detections_by_frame[row["frame_index"]].append(row)
        first_detection = next(
            (
                row
                for detections in detections_by_frame.values()
                for row in detections
            ),
            None,
        )
        detector_model_id = (
            first_detection["model_id"] if first_detection else (
                OMDET_TURBO_MODEL_ID
                if _detector_backend(workspace) == "omdet_turbo"
                else GROUNDING_DINO_MODEL_ID
            )
        )
        detector_model_revision = (
            first_detection["model_revision"] if first_detection else (
                OMDET_TURBO_REVISION
                if _detector_backend(workspace) == "omdet_turbo"
                else GROUNDING_DINO_REVISION
            )
        )
        for detections in detections_by_frame.values():
            detections.sort(key=lambda row: row["detection_index"])

        frame_table = pq.read_table(workspace.derived / "frame_index.parquet")
        timestamps = frame_table.column("timestamp_ns").to_pylist()
        capture = cv2.VideoCapture(str(workspace.raw_video))
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the copied video for object segmentation")
        assigner = _ObjectTrackAssigner()
        rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        frame_index = 0
        try:
            while True:
                decode_ok, frame = capture.read()
                if not decode_ok:
                    break
                timestamp_ns = timestamps[frame_index] if frame_index < len(timestamps) else 0
                detections = detections_by_frame.get(frame_index, [])
                segmented_count = 0
                if detections:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    boxes = [row["bbox_xyxy_px"] for row in detections]
                    inputs = processor(
                        images=image,
                        input_boxes=[boxes],
                        return_tensors="pt",
                    ).to(device)
                    with torch.inference_mode():
                        outputs = model(**inputs, multimask_output=False)
                    masks = processor.post_process_masks(
                        outputs.pred_masks.cpu(), inputs["original_sizes"]
                    )[0]
                    scores = outputs.iou_scores.detach().cpu().reshape(-1).tolist()
                    track_ids = assigner.assign(frame_index, detections)
                    height, width = frame.shape[:2]
                    for index, (detection, track_id) in enumerate(zip(detections, track_ids)):
                        mask = np.asarray(masks[index, 0], dtype=bool)
                        ys, xs = np.nonzero(mask)
                        area = int(mask.sum())
                        centroid = (
                            [float(xs.mean()), float(ys.mean())] if area > 0 else None
                        )
                        rows.append(
                            {
                                "frame_index": frame_index,
                                "timestamp_ns": timestamp_ns,
                                "detection_index": detection["detection_index"],
                                "track_id": track_id,
                                "label": detection["label"],
                                "detection_confidence": detection["confidence"],
                                "segmentation_confidence": (
                                    float(scores[index]) if index < len(scores) else None
                                ),
                                "bbox_xyxy_px": detection["bbox_xyxy_px"],
                                "mask_height": height,
                                "mask_width": width,
                                "mask_rle_counts": encode_binary_mask_rle(mask),
                                "mask_area_px": area,
                                "centroid_xy_px": centroid,
                                "observation_state": "observed",
                                "provenance": ProvenanceClass.OFFLINE_ESTIMATED.value,
                                "epistemic_status": EpistemicStatus.ESTIMATE.value,
                                "processor_id": self.processor_id,
                                "detector_model_id": detector_model_id,
                                "detector_model_revision": detector_model_revision,
                                "segmenter_model_id": SAM2_MODEL_ID,
                                "segmenter_model_revision": SAM2_REVISION,
                            }
                        )
                        segmented_count += 1
                frame_rows.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_ns": timestamp_ns,
                        "decode_ok": True,
                        "detection_count": len(detections),
                        "segmented_count": segmented_count,
                        "processor_id": self.processor_id,
                        "segmenter_model_id": SAM2_MODEL_ID,
                        "segmenter_model_revision": SAM2_REVISION,
                    }
                )
                frame_index += 1
        finally:
            capture.release()
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        for missing_index in range(frame_index, len(timestamps)):
            frame_rows.append(
                {
                    "frame_index": missing_index,
                    "timestamp_ns": timestamps[missing_index],
                    "decode_ok": False,
                    "detection_count": len(detections_by_frame.get(missing_index, [])),
                    "segmented_count": 0,
                    "processor_id": self.processor_id,
                    "segmenter_model_id": SAM2_MODEL_ID,
                    "segmenter_model_revision": SAM2_REVISION,
                }
            )

        regions_path = workspace.derived / "regions.parquet"
        frames_path = workspace.derived / "object_frames.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=REGIONS_SCHEMA), regions_path, compression="zstd"
        )
        pq.write_table(
            pa.Table.from_pylist(frame_rows, schema=OBJECT_FRAMES_SCHEMA),
            frames_path,
            compression="zstd",
        )
        return ProcessorResult(
            outputs=[regions_path, frames_path],
            metrics={
                "frames_processed": frame_index,
                "segmented_observations": len(rows),
                "frames_with_regions": sum(row["segmented_count"] > 0 for row in frame_rows),
                "unique_track_ids": len({row["track_id"] for row in rows}),
                "mask_encoding": "uncompressed COCO RLE, column-major",
                "model_id": SAM2_MODEL_ID,
                "model_revision": SAM2_REVISION,
                "detector_model_id": detector_model_id,
                "detector_model_revision": detector_model_revision,
                "device": device,
            },
        )
