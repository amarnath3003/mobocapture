from __future__ import annotations

import pyarrow as pa


FRAME_INDEX_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("source_timestamp_ns", pa.int64(), nullable=True),
        pa.field("pts", pa.int64(), nullable=True),
        pa.field("dts", pa.int64(), nullable=True),
        pa.field("duration_ns", pa.int64(), nullable=True),
        pa.field("keyframe", pa.bool_(), nullable=False),
        pa.field("picture_type", pa.string(), nullable=True),
        pa.field("packet_position", pa.int64(), nullable=True),
        pa.field("packet_size_bytes", pa.int64(), nullable=True),
        pa.field("width", pa.int32(), nullable=False),
        pa.field("height", pa.int32(), nullable=False),
        pa.field("timestamp_duplicate", pa.bool_(), nullable=False),
        pa.field("timestamp_gap", pa.bool_(), nullable=False),
        pa.field("provenance", pa.string(), nullable=False),
        pa.field("epistemic_status", pa.string(), nullable=False),
        pa.field("processor_id", pa.string(), nullable=False),
    ],
    metadata={b"mobocapture.schema": b"frame_index", b"mobocapture.version": b"0.1.0"},
)


QUALITY_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("decode_ok", pa.bool_(), nullable=False),
        pa.field("brightness_mean", pa.float32(), nullable=True),
        pa.field("underexposed_fraction", pa.float32(), nullable=True),
        pa.field("overexposed_fraction", pa.float32(), nullable=True),
        pa.field("laplacian_variance", pa.float32(), nullable=True),
        pa.field("blur_likelihood", pa.float32(), nullable=True),
        pa.field("frame_delta_mean", pa.float32(), nullable=True),
        pa.field("exact_duplicate", pa.bool_(), nullable=False),
        pa.field("near_duplicate", pa.bool_(), nullable=False),
        pa.field("camera_motion_category", pa.string(), nullable=True),
        pa.field("pixel_stats_provenance", pa.string(), nullable=False),
        pa.field("pixel_stats_status", pa.string(), nullable=False),
        pa.field("blur_status", pa.string(), nullable=False),
        pa.field("processor_id", pa.string(), nullable=False),
    ],
    metadata={b"mobocapture.schema": b"quality", b"mobocapture.version": b"0.1.0"},
)


HANDS_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("detection_index", pa.int32(), nullable=False),
        pa.field("track_id", pa.string(), nullable=False),
        pa.field("side", pa.string(), nullable=False),
        pa.field("side_confidence", pa.float32(), nullable=False),
        pa.field("bbox_xyxy_px", pa.list_(pa.float32(), 4), nullable=False),
        pa.field(
            "landmarks_2d_px",
            pa.list_(pa.list_(pa.float32(), 2), 21),
            nullable=False,
        ),
        pa.field(
            "landmarks_relative_3d",
            pa.list_(pa.list_(pa.float32(), 3), 21),
            nullable=True,
        ),
        pa.field("landmark_confidence", pa.list_(pa.float32(), 21), nullable=True),
        pa.field("visible_fraction", pa.float32(), nullable=False),
        pa.field("occluded", pa.bool_(), nullable=True),
        pa.field("truncated_by_frame", pa.bool_(), nullable=False),
        pa.field("observation_state", pa.string(), nullable=False),
        pa.field("provenance", pa.string(), nullable=False),
        pa.field("epistemic_status", pa.string(), nullable=False),
        pa.field("processor_id", pa.string(), nullable=False),
        pa.field("model_sha256", pa.string(), nullable=False),
    ],
    metadata={b"mobocapture.schema": b"hands", b"mobocapture.version": b"0.1.0"},
)


HAND_FRAMES_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("inference_timestamp_ms", pa.int64(), nullable=True),
        pa.field("decode_ok", pa.bool_(), nullable=False),
        pa.field("hand_count", pa.int32(), nullable=False),
        pa.field("left_count", pa.int32(), nullable=False),
        pa.field("right_count", pa.int32(), nullable=False),
        pa.field("unknown_count", pa.int32(), nullable=False),
        pa.field("processor_id", pa.string(), nullable=False),
        pa.field("model_sha256", pa.string(), nullable=False),
    ],
    metadata={b"mobocapture.schema": b"hand_frames", b"mobocapture.version": b"0.1.0"},
)


OBJECT_DETECTIONS_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("detection_index", pa.int32(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("confidence", pa.float32(), nullable=False),
        pa.field("bbox_xyxy_px", pa.list_(pa.float32(), 4), nullable=False),
        pa.field("prompt_concepts", pa.list_(pa.string()), nullable=False),
        pa.field("provenance", pa.string(), nullable=False),
        pa.field("epistemic_status", pa.string(), nullable=False),
        pa.field("processor_id", pa.string(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
    ],
    metadata={
        b"mobocapture.schema": b"object_detections",
        b"mobocapture.version": b"0.1.0",
    },
)


OBJECT_DETECTION_FRAMES_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("decode_ok", pa.bool_(), nullable=False),
        pa.field("detection_count", pa.int32(), nullable=False),
        pa.field("processor_id", pa.string(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
    ],
    metadata={
        b"mobocapture.schema": b"object_detection_frames",
        b"mobocapture.version": b"0.1.0",
    },
)


REGIONS_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("detection_index", pa.int32(), nullable=False),
        pa.field("track_id", pa.string(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("detection_confidence", pa.float32(), nullable=False),
        pa.field("segmentation_confidence", pa.float32(), nullable=True),
        pa.field("bbox_xyxy_px", pa.list_(pa.float32(), 4), nullable=False),
        pa.field("mask_height", pa.int32(), nullable=False),
        pa.field("mask_width", pa.int32(), nullable=False),
        pa.field("mask_rle_counts", pa.list_(pa.int32()), nullable=False),
        pa.field("mask_area_px", pa.int64(), nullable=False),
        pa.field("centroid_xy_px", pa.list_(pa.float32(), 2), nullable=True),
        pa.field("observation_state", pa.string(), nullable=False),
        pa.field("provenance", pa.string(), nullable=False),
        pa.field("epistemic_status", pa.string(), nullable=False),
        pa.field("processor_id", pa.string(), nullable=False),
        pa.field("detector_model_id", pa.string(), nullable=False),
        pa.field("detector_model_revision", pa.string(), nullable=False),
        pa.field("segmenter_model_id", pa.string(), nullable=False),
        pa.field("segmenter_model_revision", pa.string(), nullable=False),
    ],
    metadata={b"mobocapture.schema": b"regions", b"mobocapture.version": b"0.1.0"},
)


OBJECT_FRAMES_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("decode_ok", pa.bool_(), nullable=False),
        pa.field("detection_count", pa.int32(), nullable=False),
        pa.field("segmented_count", pa.int32(), nullable=False),
        pa.field("processor_id", pa.string(), nullable=False),
        pa.field("segmenter_model_id", pa.string(), nullable=False),
        pa.field("segmenter_model_revision", pa.string(), nullable=False),
    ],
    metadata={b"mobocapture.schema": b"object_frames", b"mobocapture.version": b"0.1.0"},
)


POINT_TRACKS_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("track_id", pa.string(), nullable=False),
        pa.field("xy_px", pa.list_(pa.float32(), 2), nullable=False),
        pa.field("previous_xy_px", pa.list_(pa.float32()), nullable=True),
        pa.field("displacement_xy_px", pa.list_(pa.float32()), nullable=True),
        pa.field("velocity_xy_px_s", pa.list_(pa.float32()), nullable=True),
        pa.field("tracking_error", pa.float32(), nullable=True),
        pa.field("forward_backward_error_px", pa.float32(), nullable=True),
        pa.field("observation_state", pa.string(), nullable=False),
        pa.field("provenance", pa.string(), nullable=False),
        pa.field("epistemic_status", pa.string(), nullable=False),
        pa.field("processor_id", pa.string(), nullable=False),
        pa.field("algorithm", pa.string(), nullable=False),
    ],
    metadata={b"mobocapture.schema": b"point_tracks", b"mobocapture.version": b"0.1.0"},
)


POINT_FRAMES_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("decode_ok", pa.bool_(), nullable=False),
        pa.field("visible_point_count", pa.int32(), nullable=False),
        pa.field("tracked_point_count", pa.int32(), nullable=False),
        pa.field("seeded_point_count", pa.int32(), nullable=False),
        pa.field("lost_point_count", pa.int32(), nullable=False),
        pa.field("processor_id", pa.string(), nullable=False),
        pa.field("algorithm", pa.string(), nullable=False),
    ],
    metadata={b"mobocapture.schema": b"point_frames", b"mobocapture.version": b"0.1.0"},
)


OPTICAL_FLOW_SCHEMA = pa.schema(
    [
        pa.field("source_frame_index", pa.int64(), nullable=False),
        pa.field("target_frame_index", pa.int64(), nullable=False),
        pa.field("source_timestamp_ns", pa.int64(), nullable=False),
        pa.field("target_timestamp_ns", pa.int64(), nullable=False),
        pa.field("delta_time_ns", pa.int64(), nullable=False),
        pa.field("flow_path", pa.string(), nullable=False),
        pa.field("width", pa.int32(), nullable=False),
        pa.field("height", pa.int32(), nullable=False),
        pa.field("component_order", pa.string(), nullable=False),
        pa.field("dtype", pa.string(), nullable=False),
        pa.field("valid_fraction", pa.float32(), nullable=False),
        pa.field("mean_magnitude_px", pa.float32(), nullable=False),
        pa.field("median_magnitude_px", pa.float32(), nullable=False),
        pa.field("p95_magnitude_px", pa.float32(), nullable=False),
        pa.field("maximum_magnitude_px", pa.float32(), nullable=False),
        pa.field("provenance", pa.string(), nullable=False),
        pa.field("epistemic_status", pa.string(), nullable=False),
        pa.field("processor_id", pa.string(), nullable=False),
        pa.field("algorithm", pa.string(), nullable=False),
        pa.field("parameters_json", pa.string(), nullable=False),
    ],
    metadata={b"mobocapture.schema": b"optical_flow", b"mobocapture.version": b"0.1.0"},
)


INTERACTIONS_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("hand_track_id", pa.string(), nullable=False),
        pa.field("hand_side", pa.string(), nullable=False),
        pa.field("object_track_id", pa.string(), nullable=False),
        pa.field("object_label", pa.string(), nullable=False),
        pa.field("minimum_fingertip_distance_px", pa.float32(), nullable=False),
        pa.field("normalized_fingertip_distance", pa.float32(), nullable=False),
        pa.field("fingertips_inside_mask", pa.int32(), nullable=False),
        pa.field("hand_bbox_overlap_fraction", pa.float32(), nullable=False),
        pa.field("motion_similarity", pa.float32(), nullable=True),
        pa.field("contact_likelihood", pa.float32(), nullable=False),
        pa.field("assignment_confidence", pa.float32(), nullable=False),
        pa.field("assigned_to_hand", pa.bool_(), nullable=False),
        pa.field("interaction_state", pa.string(), nullable=False),
        pa.field("event_candidate", pa.string(), nullable=True),
        pa.field("provenance", pa.string(), nullable=False),
        pa.field("epistemic_status", pa.string(), nullable=False),
        pa.field("processor_id", pa.string(), nullable=False),
        pa.field("method", pa.string(), nullable=False),
    ],
    metadata={b"mobocapture.schema": b"interactions", b"mobocapture.version": b"0.1.0"},
)


INTERACTION_FRAMES_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("hand_count", pa.int32(), nullable=False),
        pa.field("object_count", pa.int32(), nullable=False),
        pa.field("candidate_pair_count", pa.int32(), nullable=False),
        pa.field("assigned_pair_count", pa.int32(), nullable=False),
        pa.field("processor_id", pa.string(), nullable=False),
    ],
    metadata={
        b"mobocapture.schema": b"interaction_frames",
        b"mobocapture.version": b"0.1.0",
    },
)


PRIVACY_REGIONS_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("region_index", pa.int32(), nullable=False),
        pa.field("track_id", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("confidence", pa.float32(), nullable=False),
        pa.field("bbox_xyxy_px", pa.list_(pa.float32(), 4), nullable=False),
        pa.field("redaction_required", pa.bool_(), nullable=False),
        pa.field("review_required", pa.bool_(), nullable=False),
        pa.field("provenance", pa.string(), nullable=False),
        pa.field("epistemic_status", pa.string(), nullable=False),
        pa.field("processor_id", pa.string(), nullable=False),
        pa.field("detector", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
    ],
    metadata={b"mobocapture.schema": b"privacy_regions", b"mobocapture.version": b"0.1.0"},
)


PRIVACY_FRAMES_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("decode_ok", pa.bool_(), nullable=False),
        pa.field("face_count", pa.int32(), nullable=False),
        pa.field("other_privacy_count", pa.int32(), nullable=False),
        pa.field("redaction_region_count", pa.int32(), nullable=False),
        pa.field("processor_id", pa.string(), nullable=False),
    ],
    metadata={b"mobocapture.schema": b"privacy_frames", b"mobocapture.version": b"0.1.0"},
)
