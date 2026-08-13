import numpy as np

from mobocapture.modules import resolve_modules
from mobocapture.processors.privacy import _expanded_box
from mobocapture.schemas import PRIVACY_REGIONS_SCHEMA


def test_privacy_module_is_ready_and_redaction_follows_scan():
    resolved = resolve_modules(["privacy_redaction"])
    statuses = {item.processor_id: item.status for item in resolved.processors}
    assert statuses["privacy_scanner"] == "ready"
    assert statuses["privacy_redactor"] == "ready"
    ids = [item.processor_id for item in resolved.processors]
    assert ids.index("privacy_scanner") < ids.index("privacy_redactor")


def test_redaction_box_expands_and_clamps_to_frame():
    assert _expanded_box([10, 10, 30, 30], 100, 100) == (7, 7, 33, 33)
    assert _expanded_box([0, 0, 20, 20], 25, 25) == (0, 0, 23, 23)


def test_privacy_schema_is_auditable_without_identity_fields():
    names = set(PRIVACY_REGIONS_SCHEMA.names)
    assert {"category", "confidence", "redaction_required", "review_required"}.issubset(names)
    assert not any("identity" in name or "name" in name for name in names)
