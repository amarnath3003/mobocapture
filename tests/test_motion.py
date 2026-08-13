import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from mobocapture.modules import resolve_modules
from mobocapture.pipeline import process_video
from mobocapture.processors.motion import _save_npz_lossless_fast


def test_motion_module_is_ready():
    resolved = resolve_modules(["motion_tracking"])
    statuses = {item.processor_id: item.status for item in resolved.processors}
    assert statuses["point_tracker"] == "ready"
    assert "optical_flow" not in statuses


def test_fast_npz_is_bit_exact_and_numpy_compatible(tmp_path: Path):
    expected = np.arange(96, dtype=np.float16).reshape(4, 6, 4)
    destination = tmp_path / "flow.npz"
    _save_npz_lossless_fast(destination, flow_xy_px=expected)
    with np.load(destination) as payload:
        np.testing.assert_array_equal(payload["flow_xy_px"], expected)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg tools are required",
)
def test_motion_pipeline_writes_tracks_dense_arrays_and_overlay(tmp_path: Path):
    video = tmp_path / "motion.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=10:duration=0.8",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    workspace = process_video(
        video,
        tmp_path / "session",
        module_ids=["motion_tracking", "dense_optical_flow"],
    )
    points = pq.read_table(workspace.derived / "point_tracks.parquet").to_pylist()
    point_frames = pq.read_table(workspace.derived / "point_frames.parquet").to_pylist()
    flow_rows = pq.read_table(workspace.derived / "optical_flow.parquet").to_pylist()
    assert len(point_frames) == 8
    assert any(row["observation_state"] == "tracked" for row in points)
    assert len(flow_rows) == 7
    flow_path = workspace.root / flow_rows[0]["flow_path"]
    with np.load(flow_path) as payload:
        assert payload["flow_xy_px"].shape == (120, 160, 2)
    assert (workspace.review / "overlay.mp4").stat().st_size > 0
    assert workspace.manifest.status == "complete"
