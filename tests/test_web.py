import io
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mobocapture.web import create_app


def _video_bytes(tmp_path: Path) -> bytes:
    video = tmp_path / "web-input.mp4"
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
    return video.read_bytes()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg tools are required",
)
def test_web_upload_process_preview_and_download(tmp_path: Path):
    app = create_app(tmp_path / "web-data")
    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert '<div id="root"></div>' in index.text
        assert "/static/assets/" in index.text
        assert "React" not in index.text  # React renders into the empty production shell.

        catalog = client.get("/api/modules").json()
        assert catalog["modules"][0]["id"] == "video_quality"
        assert catalog["modules"][0]["status"] == "ready"
        hands = next(item for item in catalog["modules"] if item["id"] == "hands_fingers")
        assert hands["status"] == "ready"
        objects = next(item for item in catalog["modules"] if item["id"] == "objects")
        assert objects["status"] == "ready"

        response = client.post(
            "/api/jobs",
            files={"video": ("demo.mp4", _video_bytes(tmp_path), "video/mp4")},
            data={"modules": "video_quality"},
        )
        assert response.status_code == 202
        job = response.json()
        deadline = time.monotonic() + 15
        while job["status"] not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.05)
            job = client.get(f"/api/jobs/{job['job_id']}").json()

        assert job["status"] == "completed", job.get("error")
        assert job["progress"] == 100
        assert job["session_status"] == "complete"
        assert job["completed_processors"] == [
            "video_ingest",
            "video_quality",
            "overlay_renderer",
        ]

        overlay = client.get(job["overlay_url"])
        assert overlay.status_code == 200
        assert overlay.headers["content-type"].startswith("video/mp4")
        assert len(overlay.content) > 100

        download = client.get(job["download_url"])
        assert download.status_code == 200
        with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
            names = set(archive.namelist())
            assert "raw/video.mp4" in names
            assert "review/overlay.mp4" in names
            assert "derived/v0.1/frame_index.parquet" in names
            assert "derived/v0.1/quality.parquet" in names
            session = json.loads(archive.read("manifests/session.json"))
            assert session["status"] == "complete"


def test_web_rejects_empty_upload(tmp_path: Path):
    app = create_app(tmp_path / "web-data")
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            files={"video": ("empty.mp4", b"", "video/mp4")},
            data={"modules": "video_quality"},
        )
        assert response.status_code == 422
