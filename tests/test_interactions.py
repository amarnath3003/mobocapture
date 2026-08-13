import numpy as np

from mobocapture.models import EpistemicStatus
from mobocapture.modules import resolve_modules
from mobocapture.processors.interactions import _motion_similarity, _overlap_fraction
from mobocapture.schemas import INTERACTIONS_SCHEMA


def test_interaction_module_is_ready_with_all_visual_dependencies():
    resolved = resolve_modules(["hand_object_interactions"])
    statuses = {item.processor_id: item.status for item in resolved.processors}
    assert statuses["hand_tracker"] == "ready"
    assert statuses["object_segmenter"] == "ready"
    assert statuses["point_tracker"] == "ready"
    assert statuses["interaction_inference"] == "ready"


def test_interaction_evidence_features_are_bounded():
    assert _overlap_fraction([0, 0, 10, 10], [5, 0, 15, 10]) == 0.5
    assert _motion_similarity(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == 1.0
    assert _motion_similarity(np.array([1.0, 0.0]), np.array([-2.0, 0.0])) == 0.0
    assert _motion_similarity(np.zeros(2), np.ones(2)) is None


def test_interaction_schema_marks_hypotheses_separately_from_provenance():
    assert "epistemic_status" in INTERACTIONS_SCHEMA.names
    assert "provenance" in INTERACTIONS_SCHEMA.names
    assert "minimum_fingertip_distance_px" in INTERACTIONS_SCHEMA.names
    assert EpistemicStatus.HYPOTHESIS.value == "hypothesis"
