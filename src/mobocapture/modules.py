from __future__ import annotations

from dataclasses import dataclass

from mobocapture import __version__
from mobocapture.models import ResolvedProcessor, ResolvedProcessorsManifest


@dataclass(frozen=True)
class ModuleDefinition:
    module_id: str
    label: str
    description: str
    dependencies: tuple[str, ...]
    processors: tuple[str, ...]
    cost_tier: str = "standard"
    storage_note: str = "Compact tabular output"
    warning: str | None = None


@dataclass(frozen=True)
class ProcessorDefinition:
    processor_id: str
    requires: tuple[str, ...]
    produces: tuple[str, ...]
    implemented: bool
    code_license: str = "NOASSERTION (project license not yet selected)"
    model_license: str | None = None
    unavailable_reason: str | None = None


PROCESSORS: dict[str, ProcessorDefinition] = {
    "video_ingest": ProcessorDefinition(
        "video_ingest",
        (),
        ("video.container_metadata", "video.frame_index", "video.timestamp_health"),
        True,
    ),
    "video_quality": ProcessorDefinition(
        "video_quality",
        ("video_ingest",),
        (
            "video.decode_health",
            "video.duplicates",
            "video.near_duplicates",
            "quality.blur",
            "quality.exposure",
            "quality.camera_motion",
        ),
        True,
    ),
    "overlay_renderer": ProcessorDefinition(
        "overlay_renderer", ("video_quality",), ("review.overlay_video",), True
    ),
    "hand_tracker": ProcessorDefinition(
        "hand_tracker",
        ("video_ingest",),
        ("hand.detection", "hand.tracks", "hand.keypoints_2d", "hand.fingertips"),
        True,
        code_license="Apache-2.0 (MediaPipe runtime)",
        model_license="Apache-2.0 official MediaPipe task bundle",
    ),
    "object_detector": ProcessorDefinition(
        "object_detector",
        ("video_ingest",),
        ("object.open_vocab_detection",),
        True,
        code_license="Apache-2.0 (Transformers/OmDet-Turbo runtime)",
        model_license=(
            "Apache-2.0 (omlab/omdet-turbo-swin-tiny-hf default; "
            "Grounding DINO compatibility backend, pinned revisions)"
        ),
    ),
    "object_segmenter": ProcessorDefinition(
        "object_segmenter",
        ("object_detector",),
        ("object.open_vocab_masks", "object.instance_tracks"),
        True,
        code_license="Apache-2.0 (Transformers/SAM2 runtime)",
        model_license="Apache-2.0 (facebook/sam2.1-hiera-tiny, pinned revision)",
    ),
    "point_tracker": ProcessorDefinition(
        "point_tracker",
        ("video_ingest",),
        ("motion.point_tracks_2d",),
        True,
        code_license="Apache-2.0 (OpenCV Shi-Tomasi/Lucas-Kanade runtime)",
        model_license=None,
    ),
    "optical_flow": ProcessorDefinition(
        "optical_flow",
        ("video_ingest",),
        ("motion.optical_flow",),
        True,
        code_license="Apache-2.0 (OpenCV Farneback runtime)",
        model_license=None,
    ),
    "interaction_inference": ProcessorDefinition(
        "interaction_inference",
        ("hand_tracker", "object_segmenter", "point_tracker"),
        (
            "interaction.hand_object_assignment",
            "interaction.contact_likelihood",
            "interaction.grasp",
            "interaction.release",
            "interaction.bimanual_roles",
        ),
        True,
        code_license="NOASSERTION (MoboCapture evidence heuristic; project license pending)",
    ),
    "temporal_segmenter": ProcessorDefinition(
        "temporal_segmenter",
        ("video_quality", "hand_tracker", "object_segmenter", "point_tracker"),
        ("temporal.event_boundaries", "temporal.action_segments"),
        False,
        unavailable_reason="Evidence-driven temporal segmentation is not integrated yet",
    ),
    "vlm_describer": ProcessorDefinition(
        "vlm_describer",
        ("temporal_segmenter",),
        (
            "semantic.segment_description",
            "semantic.dense_steps",
            "semantic.goal",
            "semantic.action_triplets",
            "semantic.outcome",
        ),
        False,
        model_license="weight-specific; pending audit",
        unavailable_reason="Local schema-constrained Qwen3-VL adapter is not integrated yet",
    ),
    "privacy_scanner": ProcessorDefinition(
        "privacy_scanner",
        ("video_ingest",),
        (
            "privacy.faces",
            "privacy.plates",
            "privacy.screens",
            "privacy.documents",
            "privacy.reflections",
        ),
        True,
        code_license="Apache-2.0/MIT (OpenCV, Transformers, YuNet adapter)",
        model_license=(
            "MIT (OpenCV Zoo YuNet) + Apache-2.0 "
            "(IDEA-Research/grounding-dino-tiny, pinned revision)"
        ),
    ),
    "privacy_redactor": ProcessorDefinition(
        "privacy_redactor",
        ("privacy_scanner",),
        ("privacy.redaction",),
        True,
        code_license="NOASSERTION (MoboCapture/FFmpeg redaction adapter; project license pending)",
    ),
    "rgb_geometry": ProcessorDefinition(
        "rgb_geometry",
        ("video_ingest", "point_tracker"),
        (
            "geometry.monocular_depth",
            "camera.pose_relative",
            "geometry.pointcloud_relative",
        ),
        False,
        model_license="Apache-2.0 candidate; exact weights pending audit",
        unavailable_reason="Experimental Depth Anything 3 geometry is intentionally deferred",
    ),
}


MODULES: dict[str, ModuleDefinition] = {
    "video_quality": ModuleDefinition(
        "video_quality",
        "Video quality",
        "Timing, decode health, exposure, blur, duplicates, motion, and overlay",
        (),
        ("video_ingest", "video_quality", "overlay_renderer"),
    ),
    "hands_fingers": ModuleDefinition(
        "hands_fingers",
        "Hands & fingers",
        "Persistent left/right hand tracks and 21 landmarks",
        (),
        ("hand_tracker",),
    ),
    "objects": ModuleDefinition(
        "objects",
        "Objects",
        "Open-vocabulary boxes, masks, and persistent visual tracks",
        (),
        ("object_detector", "object_segmenter"),
    ),
    "motion_tracking": ModuleDefinition(
        "motion_tracking",
        "Sparse motion tracks",
        "Persistent point trajectories for robot motion and interaction evidence",
        (),
        ("point_tracker",),
        cost_tier="standard",
        storage_note="Compact Parquet trajectories; typically tens of MB per minute",
    ),
    "dense_optical_flow": ModuleDefinition(
        "dense_optical_flow",
        "Dense optical flow (advanced)",
        "A full-resolution 2D motion vector for every pixel and adjacent frame pair",
        ("motion_tracking",),
        ("optical_flow",),
        cost_tier="extreme",
        storage_note="About 13 GB/min at 1080p30 before archive overhead",
        warning=(
            "Expert-only dense raster product. It is slow and very large; sparse "
            "motion tracks are the robot-learning default."
        ),
    ),
    "hand_object_interactions": ModuleDefinition(
        "hand_object_interactions",
        "Hand-object interactions",
        "Associations, contact likelihood, grasp/release, and bimanual roles",
        ("hands_fingers", "objects", "motion_tracking"),
        ("interaction_inference",),
    ),
    "vlm_descriptions": ModuleDefinition(
        "vlm_descriptions",
        "VLM descriptions",
        "Evidence-linked temporal steps, objects, hand roles, and outcomes",
        ("video_quality", "hands_fingers", "objects", "motion_tracking"),
        ("temporal_segmenter", "vlm_describer"),
    ),
    "privacy_redaction": ModuleDefinition(
        "privacy_redaction",
        "Privacy & redaction",
        "Privacy candidates and a governed sanitized derivative",
        (),
        ("privacy_scanner", "privacy_redactor"),
    ),
    "rgb_geometry_experimental": ModuleDefinition(
        "rgb_geometry_experimental",
        "RGB geometry (experimental)",
        "Relative depth, camera motion, and relative-scale geometry",
        ("motion_tracking",),
        ("rgb_geometry",),
    ),
}


PROFILES: dict[str, tuple[str, ...]] = {
    "foundation": ("video_quality",),
    "rgb_core": (
        "video_quality",
        "hands_fingers",
        "objects",
        "motion_tracking",
        "privacy_redaction",
    ),
    "interaction_max": (
        "video_quality",
        "hands_fingers",
        "objects",
        "motion_tracking",
        "hand_object_interactions",
        "privacy_redaction",
    ),
    "geometry_experimental": (
        "video_quality",
        "motion_tracking",
        "rgb_geometry_experimental",
    ),
    # Safe comprehensive preset: excludes unavailable research modules and the
    # expert-only dense raster flow product.
    "full_rgb": (
        "video_quality",
        "hands_fingers",
        "objects",
        "motion_tracking",
        "hand_object_interactions",
        "privacy_redaction",
    ),
    "archival_dense": (
        "video_quality",
        "hands_fingers",
        "objects",
        "motion_tracking",
        "dense_optical_flow",
        "hand_object_interactions",
        "privacy_redaction",
    ),
}


def selected_modules(module_ids: list[str] | None, profile: str | None) -> list[str]:
    selected: list[str] = []
    if profile:
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile '{profile}'. Available: {', '.join(PROFILES)}")
        selected.extend(PROFILES[profile])
    if module_ids:
        selected.extend(module_ids)
    if not selected:
        selected.extend(PROFILES["foundation"])
    unknown = sorted(set(selected) - MODULES.keys())
    if unknown:
        raise ValueError(f"Unknown module(s): {', '.join(unknown)}")
    return list(dict.fromkeys(selected))


def resolve_modules(module_ids: list[str]) -> ResolvedProcessorsManifest:
    resolved_modules: list[str] = []

    def add_module(module_id: str) -> None:
        if module_id in resolved_modules:
            return
        for dependency in MODULES[module_id].dependencies:
            add_module(dependency)
        resolved_modules.append(module_id)

    # Ingest, timing, quality, and the review artifact are foundational for
    # every run, even when the user selects only a learned capability.
    add_module("video_quality")
    for module_id in module_ids:
        add_module(module_id)

    processor_ids: list[str] = []

    def add_processor(processor_id: str) -> None:
        if processor_id in processor_ids:
            return
        for dependency in PROCESSORS[processor_id].requires:
            add_processor(dependency)
        processor_ids.append(processor_id)

    for module_id in resolved_modules:
        for processor_id in MODULES[module_id].processors:
            add_processor(processor_id)

    resolved: list[ResolvedProcessor] = []
    for processor_id in processor_ids:
        definition = PROCESSORS[processor_id]
        required_by = [
            module_id
            for module_id in resolved_modules
            if processor_id in MODULES[module_id].processors
        ]
        resolved.append(
            ResolvedProcessor(
                processor_id=processor_id,
                version=__version__,
                status="ready" if definition.implemented else "unavailable",
                required_by_modules=required_by,
                requires=list(definition.requires),
                produces=list(definition.produces),
                code_license=definition.code_license,
                model_license=definition.model_license,
                unavailable_reason=definition.unavailable_reason,
            )
        )

    return ResolvedProcessorsManifest(
        requested_modules=module_ids,
        resolved_modules=resolved_modules,
        processors=resolved,
    )
