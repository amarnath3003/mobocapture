import numpy as np

from mobocapture.modules import resolve_modules
from mobocapture.processors.objects import (
    _ObjectTrackAssigner,
    decode_binary_mask_rle,
    encode_binary_mask_rle,
)
from mobocapture.schemas import REGIONS_SCHEMA


def test_objects_module_is_ready_and_overlay_runs_after_it():
    resolved = resolve_modules(["objects"])
    statuses = {item.processor_id: item.status for item in resolved.processors}
    assert statuses["object_detector"] == "ready"
    assert statuses["object_segmenter"] == "ready"
    assert [item.processor_id for item in resolved.processors].index(
        "object_detector"
    ) < [item.processor_id for item in resolved.processors].index("object_segmenter")


def test_object_track_assigner_persists_ids_by_label_and_overlap():
    assigner = _ObjectTrackAssigner()
    first = assigner.assign(
        0,
        [
            {"label": "cup", "bbox_xyxy_px": [10, 10, 40, 40]},
            {"label": "bottle", "bbox_xyxy_px": [60, 10, 90, 60]},
        ],
    )
    second = assigner.assign(
        1,
        [
            {"label": "cup", "bbox_xyxy_px": [12, 11, 42, 41]},
            {"label": "bottle", "bbox_xyxy_px": [62, 10, 92, 60]},
        ],
    )
    assert first == second == ["object-0001", "object-0002"]


def test_mask_rle_round_trip_and_regions_truth_fields():
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 2:6] = True
    counts = encode_binary_mask_rle(mask)
    np.testing.assert_array_equal(decode_binary_mask_rle(counts, 5, 7), mask)
    names = set(REGIONS_SCHEMA.names)
    assert {
        "track_id",
        "label",
        "detection_confidence",
        "segmentation_confidence",
        "mask_rle_counts",
        "provenance",
        "epistemic_status",
        "detector_model_revision",
        "segmenter_model_revision",
    }.issubset(names)
