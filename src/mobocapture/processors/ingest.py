from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from mobocapture.models import EpistemicStatus, ProvenanceClass, VideoMetadata
from mobocapture.processors.base import Processor, ProcessorResult
from mobocapture.schemas import FRAME_INDEX_SCHEMA
from mobocapture.session import SessionWorkspace


def _require_ffprobe() -> str:
    executable = shutil.which("ffprobe")
    if not executable:
        raise RuntimeError("ffprobe is required but was not found on PATH")
    return executable


def _ffprobe_version() -> str:
    completed = subprocess.run(
        [_require_ffprobe(), "-version"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.splitlines()[0]


def _run_ffprobe(arguments: list[str], video: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [_require_ffprobe(), "-v", "error", *arguments, "-of", "json", str(video)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def _optional_int(value: Any) -> int | None:
    if value in (None, "N/A", ""):
        return None
    return int(value)


def _seconds_to_ns(value: Any) -> int | None:
    if value in (None, "N/A", ""):
        return None
    return int(round(float(value) * 1_000_000_000))


def _timestamp_ns(frame: dict[str, Any], time_base: Fraction) -> int | None:
    timestamp = _optional_int(frame.get("best_effort_timestamp"))
    if timestamp is not None:
        return round(timestamp * time_base.numerator * 1_000_000_000 / time_base.denominator)
    return _seconds_to_ns(frame.get("best_effort_timestamp_time"))


def _rotation(stream: dict[str, Any]) -> int:
    for side_data in stream.get("side_data_list", []):
        if side_data.get("rotation") is not None:
            return int(round(float(side_data["rotation"]))) % 360
    tags = stream.get("tags", {})
    return int(round(float(tags.get("rotate", 0)))) % 360


def _fps(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    rate = Fraction(value)
    result = float(rate)
    return result if result > 0 else None


class VideoIngestProcessor(Processor):
    processor_id = "video_ingest"

    def process(self, workspace: SessionWorkspace) -> ProcessorResult:
        probe = _run_ffprobe(
            ["-select_streams", "v:0", "-show_streams", "-show_format"],
            workspace.raw_video,
        )
        streams = probe.get("streams", [])
        if not streams:
            raise ValueError("Input has no video stream")
        stream = streams[0]
        time_base_text = stream.get("time_base")
        if not time_base_text:
            raise ValueError("Video stream has no time base")
        time_base = Fraction(time_base_text)

        frames_probe = _run_ffprobe(
            [
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                (
                    "frame=best_effort_timestamp,best_effort_timestamp_time,pts,pkt_dts,"
                    "duration,duration_time,key_frame,pkt_pos,pkt_size,width,height,pict_type"
                ),
            ],
            workspace.raw_video,
        )
        frame_records = frames_probe.get("frames", [])
        if not frame_records:
            raise ValueError("Video stream contains no decodable frames")

        source_timestamps = [_timestamp_ns(frame, time_base) for frame in frame_records]
        first_timestamp = next((value for value in source_timestamps if value is not None), 0)
        normalized = [
            (value - first_timestamp) if value is not None else None for value in source_timestamps
        ]
        known_deltas = [
            current - previous
            for previous, current in zip(normalized, normalized[1:])
            if previous is not None and current is not None and current > previous
        ]
        median_delta = sorted(known_deltas)[len(known_deltas) // 2] if known_deltas else None

        rows: list[dict[str, Any]] = []
        previous_timestamp: int | None = None
        for index, (frame, source_ns, timestamp_ns) in enumerate(
            zip(frame_records, source_timestamps, normalized)
        ):
            if timestamp_ns is None:
                if previous_timestamp is None:
                    timestamp_ns = 0
                elif median_delta is not None:
                    timestamp_ns = previous_timestamp + median_delta
                else:
                    timestamp_ns = previous_timestamp
            duplicate = previous_timestamp is not None and timestamp_ns <= previous_timestamp
            gap = bool(
                previous_timestamp is not None
                and median_delta is not None
                and timestamp_ns - previous_timestamp > median_delta * 3 // 2
            )
            rows.append(
                {
                    "frame_index": index,
                    "timestamp_ns": timestamp_ns,
                    "source_timestamp_ns": source_ns,
                    "pts": _optional_int(frame.get("pts")),
                    "dts": _optional_int(frame.get("pkt_dts")),
                    "duration_ns": (
                        round(
                            _optional_int(frame.get("duration"))
                            * time_base.numerator
                            * 1_000_000_000
                            / time_base.denominator
                        )
                        if _optional_int(frame.get("duration")) is not None
                        else _seconds_to_ns(frame.get("duration_time"))
                    ),
                    "keyframe": bool(int(frame.get("key_frame", 0))),
                    "picture_type": frame.get("pict_type"),
                    "packet_position": _optional_int(frame.get("pkt_pos")),
                    "packet_size_bytes": _optional_int(frame.get("pkt_size")),
                    "width": int(frame.get("width", stream["width"])),
                    "height": int(frame.get("height", stream["height"])),
                    "timestamp_duplicate": duplicate,
                    "timestamp_gap": gap,
                    "provenance": ProvenanceClass.MEASURED.value,
                    "epistemic_status": EpistemicStatus.DETERMINISTIC.value,
                    "processor_id": self.processor_id,
                }
            )
            previous_timestamp = timestamp_ns

        frame_index_path = workspace.derived / "frame_index.parquet"
        table = pa.Table.from_pylist(rows, schema=FRAME_INDEX_SCHEMA)
        pq.write_table(table, frame_index_path, compression="zstd")

        format_data = probe.get("format", {})
        duration_ns = _seconds_to_ns(stream.get("duration")) or _seconds_to_ns(
            format_data.get("duration")
        )
        metadata = VideoMetadata(
            codec_name=stream.get("codec_name"),
            codec_long_name=stream.get("codec_long_name"),
            pixel_format=stream.get("pix_fmt"),
            width=int(stream["width"]),
            height=int(stream["height"]),
            display_rotation_degrees=_rotation(stream),
            nominal_frame_rate=_fps(stream.get("r_frame_rate")),
            average_frame_rate=stream.get("avg_frame_rate"),
            time_base=time_base_text,
            duration_ns=duration_ns,
            probed_frame_count=len(rows),
            color_range=stream.get("color_range"),
            color_space=stream.get("color_space"),
            color_transfer=stream.get("color_transfer"),
            color_primaries=stream.get("color_primaries"),
        )
        workspace.manifest.video = metadata
        workspace.write_session_manifest()

        return ProcessorResult(
            outputs=[frame_index_path],
            metrics={
                "frames": len(rows),
                "timestamp_duplicates": sum(row["timestamp_duplicate"] for row in rows),
                "timestamp_gaps": sum(row["timestamp_gap"] for row in rows),
                "decoder": _ffprobe_version(),
            },
        )
