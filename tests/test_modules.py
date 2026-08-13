from mobocapture.models import EpistemicStatus, ProvenanceClass
from mobocapture.modules import resolve_modules, selected_modules


def test_foundation_is_default_and_ready():
    selected = selected_modules(None, None)
    assert selected == ["video_quality"]
    resolved = resolve_modules(selected)
    assert [item.processor_id for item in resolved.processors] == [
        "video_ingest",
        "video_quality",
        "overlay_renderer",
    ]
    assert all(item.status == "ready" for item in resolved.processors)


def test_interactions_resolve_visual_dependencies():
    resolved = resolve_modules(["hand_object_interactions"])
    assert resolved.resolved_modules == [
        "video_quality",
        "hands_fingers",
        "objects",
        "motion_tracking",
        "hand_object_interactions",
    ]
    processor_ids = [item.processor_id for item in resolved.processors]
    assert "hand_tracker" in processor_ids
    assert "object_segmenter" in processor_ids
    assert "point_tracker" in processor_ids
    assert "interaction_inference" in processor_ids
    assert "optical_flow" not in processor_ids


def test_dense_flow_is_explicit_and_excluded_from_safe_presets():
    resolved = resolve_modules(["dense_optical_flow"])
    assert "optical_flow" in [item.processor_id for item in resolved.processors]
    assert "dense_optical_flow" not in selected_modules(None, "full_rgb")


def test_foundation_is_implicit_for_every_selection():
    resolved = resolve_modules(["hands_fingers"])
    assert resolved.resolved_modules == ["video_quality", "hands_fingers"]
    assert [item.processor_id for item in resolved.processors][:3] == [
        "video_ingest",
        "video_quality",
        "overlay_renderer",
    ]


def test_provenance_and_epistemic_status_are_independent():
    assert ProvenanceClass.OFFLINE_ESTIMATED.value == "offline_estimated"
    assert EpistemicStatus.HYPOTHESIS.value == "hypothesis"
