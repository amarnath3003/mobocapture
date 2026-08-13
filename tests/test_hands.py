from mobocapture.modules import resolve_modules
from mobocapture.processors.hands import _TrackAssigner
from mobocapture.schemas import HANDS_SCHEMA


def test_hands_module_is_ready_and_overlay_runs_after_it():
    resolved = resolve_modules(["hands_fingers"])
    processors = {item.processor_id: item for item in resolved.processors}
    assert processors["hand_tracker"].status == "ready"
    assert processors["hand_tracker"].produces == [
        "hand.detection",
        "hand.tracks",
        "hand.keypoints_2d",
        "hand.fingertips",
    ]


def test_track_assigner_persists_ids_and_separates_hands():
    assigner = _TrackAssigner()
    first = assigner.assign(
        0,
        [(0.2, 0.5, "left", 0.9), (0.8, 0.5, "right", 0.95)],
    )
    second = assigner.assign(
        1,
        [(0.22, 0.5, "left", 0.9), (0.78, 0.5, "right", 0.95)],
    )
    assert first == [("hand-0001", "left"), ("hand-0002", "right")]
    assert second == first


def test_hands_schema_keeps_confidence_and_truth_status_separate():
    names = set(HANDS_SCHEMA.names)
    assert {
        "landmarks_2d_px",
        "landmark_confidence",
        "provenance",
        "epistemic_status",
        "model_sha256",
    } <= names
