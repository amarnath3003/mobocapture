import json
import shutil
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from mobocapture.io import sha256_file
from mobocapture.pipeline import process_video


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg tools are required",
)
def test_foundation_pipeline_end_to_end(tmp_path: Path):
    video = tmp_path / "input.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=10:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    original_hash = sha256_file(video)
    session_root = tmp_path / "session"

    workspace = process_video(video, session_root)

    assert workspace.manifest.status == "complete"
    assert sha256_file(session_root / "raw" / "video.mp4") == original_hash
    frame_table = pq.read_table(session_root / "derived" / "v0.1" / "frame_index.parquet")
    quality_table = pq.read_table(session_root / "derived" / "v0.1" / "quality.parquet")
    assert frame_table.num_rows == 10
    assert quality_table.num_rows == 10
    assert frame_table.column("timestamp_ns").to_pylist()[0] == 0
    assert (session_root / "review" / "overlay.mp4").stat().st_size > 0
    provenance = json.loads(
        (session_root / "derived" / "v0.1" / "provenance.json").read_text()
    )
    assert [run["processor_id"] for run in provenance["processor_runs"]] == [
        "video_ingest",
        "video_quality",
        "overlay_renderer",
    ]

    session = json.loads((session_root / "manifests" / "session.json").read_text())
    assert session["input"]["sha256"] == original_hash
    assert session["video"]["probed_frame_count"] == 10
    assert session["status"] == "complete"


def test_nonempty_output_is_rejected(tmp_path: Path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"not used because output validation happens after input validation")
    output = tmp_path / "session"
    output.mkdir()
    (output / "user-file.txt").write_text("preserve me")
    with pytest.raises(FileExistsError):
        process_video(video, output)
    assert (output / "user-file.txt").read_text() == "preserve me"
