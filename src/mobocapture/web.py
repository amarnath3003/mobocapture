from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
import zipfile
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mobocapture.modules import MODULES, PROFILES, resolve_modules, selected_modules
from mobocapture.pipeline import process_video


JobStatus = Literal["queued", "processing", "completed", "failed"]
MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024
SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9]{1,10}$")


def _utc_text() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobState:
    job_id: str
    source_name: str
    requested_modules: list[str]
    processing_options: dict = field(default_factory=dict)
    status: JobStatus = "queued"
    progress: int = 0
    stage: str = "Waiting to start"
    created_at_utc: str = field(default_factory=_utc_text)
    updated_at_utc: str = field(default_factory=_utc_text)
    session_status: str | None = None
    session_id: str | None = None
    completed_processors: list[str] = field(default_factory=list)
    unavailable_processors: list[str] = field(default_factory=list)
    error: str | None = None
    overlay_url: str | None = None
    download_url: str | None = None
    redacted_url: str | None = None


class JobManager:
    """Owns local web jobs and serializes processing for future GPU workers."""

    def __init__(self, data_root: Path, max_workers: int = 1):
        self.data_root = data_root.resolve()
        self.upload_root = self.data_root / "uploads"
        self.session_root = self.data_root / "sessions"
        self.download_root = self.data_root / "downloads"
        for directory in (self.upload_root, self.session_root, self.download_root):
            directory.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mobocapture")

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def create(
        self,
        source_name: str,
        upload_path: Path,
        modules: list[str],
        processing_options: dict | None = None,
    ) -> JobState:
        job = JobState(
            job_id=upload_path.parent.name,
            source_name=source_name,
            requested_modules=modules,
            processing_options=processing_options or {},
        )
        with self._lock:
            self._jobs[job.job_id] = job
        self._executor.submit(self._run, job.job_id, upload_path)
        return self.get(job.job_id)

    def get(self, job_id: str) -> JobState:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return JobState(**asdict(job))

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at_utc = _utc_text()

    def _progress(self, job_id: str, stage: str, completed: int, total: int) -> None:
        labels = {
            "video_ingest": "Reading timestamps and video metadata",
            "video_quality": "Measuring frame and image quality",
            "hand_tracker": "Tracking hands and finger landmarks",
            "object_detector": "Finding prompted objects in every frame",
            "object_segmenter": "Segmenting objects and assigning track IDs",
            "point_tracker": "Tracking persistent visual points",
            "optical_flow": "Computing dense frame-to-frame motion",
            "interaction_inference": "Linking hands to objects and interaction candidates",
            "privacy_scanner": "Scanning faces, screens, documents, plates, and mirrors",
            "privacy_redactor": "Rendering the governed redacted derivative",
            "overlay_renderer": "Rendering the review overlay",
            "finalizing": "Writing manifests and provenance",
        }
        progress = 8 if total == 0 else 8 + int(82 * completed / total)
        if stage == "finalizing":
            progress = 92
        self._update(
            job_id,
            status="processing",
            stage=labels.get(stage, stage.replace("_", " ").title()),
            progress=min(progress, 95),
        )

    def _run(self, job_id: str, upload_path: Path) -> None:
        output = self.session_root / job_id
        self._update(job_id, status="processing", progress=4, stage="Preparing local session")
        try:
            job = self.get(job_id)
            workspace = process_video(
                upload_path,
                output,
                module_ids=job.requested_modules,
                options=job.processing_options,
                progress_callback=lambda stage, completed, total: self._progress(
                    job_id, stage, completed, total
                ),
            )
            run_report = json.loads((workspace.manifests / "run_report.json").read_text("utf-8"))
            self._update(job_id, progress=96, stage="Packaging complete dataset")
            archive = self.download_root / f"{job_id}.zip"
            _archive_session(workspace.root, archive)
            self._update(
                job_id,
                status="completed",
                progress=100,
                stage="Ready",
                session_status=workspace.manifest.status,
                session_id=workspace.manifest.session_id,
                completed_processors=run_report["completed_processors"],
                unavailable_processors=workspace.manifest.unavailable_processors,
                overlay_url=f"/api/jobs/{job_id}/overlay",
                download_url=f"/api/jobs/{job_id}/download",
                redacted_url=(
                    f"/api/jobs/{job_id}/redacted"
                    if (workspace.review / "redacted.mp4").is_file()
                    else None
                ),
            )
        except Exception as error:
            self._update(
                job_id,
                status="failed",
                stage="Processing failed",
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            # The canonical session contains its own hash-verified raw copy.
            upload_path.unlink(missing_ok=True)
            try:
                upload_path.parent.rmdir()
            except OSError:
                pass


def _archive_session(session_root: Path, destination: Path) -> None:
    temporary = destination.with_suffix(".zip.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for path in sorted(session_root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(session_root).as_posix()
            compression = (
                zipfile.ZIP_STORED
                if path.suffix.lower()
                in {
                    ".mp4",
                    ".mov",
                    ".mkv",
                    ".webm",
                    ".avi",
                    ".npz",
                    ".parquet",
                    ".jpg",
                    ".jpeg",
                    ".png",
                }
                else zipfile.ZIP_DEFLATED
            )
            archive.write(path, relative, compress_type=compression)
    temporary.replace(destination)


def _job_payload(job: JobState) -> dict:
    return asdict(job)


def create_app(data_root: Path | None = None, max_upload_bytes: int = MAX_UPLOAD_BYTES) -> FastAPI:
    root = data_root or Path("mobocapture-web-data")
    manager = JobManager(root)
    static_root = Path(__file__).with_name("static")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.close()

    app = FastAPI(title="MoboCapture", version="0.1.0", lifespan=lifespan)
    app.state.jobs = manager
    app.state.max_upload_bytes = max_upload_bytes
    app.mount("/static", StaticFiles(directory=static_root), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_root / "index.html")

    @app.get("/api/modules")
    def module_catalog() -> dict:
        capabilities = []
        for definition in MODULES.values():
            resolution = resolve_modules([definition.module_id])
            available = all(item.status == "ready" for item in resolution.processors)
            capabilities.append(
                {
                    "id": definition.module_id,
                    "label": definition.label,
                    "description": definition.description,
                    "status": "ready" if available else "planned",
                    "foundational": definition.module_id == "video_quality",
                    "cost_tier": definition.cost_tier,
                    "storage_note": definition.storage_note,
                    "warning": definition.warning,
                }
            )
        return {"modules": capabilities, "profiles": {key: list(value) for key, value in PROFILES.items()}}

    @app.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
    async def create_job(
        video: UploadFile = File(...),
        modules: list[str] | None = Form(default=None),
        object_concepts: str | None = Form(default=None),
    ) -> dict:
        try:
            requested = selected_modules(modules, None)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if not video.filename:
            raise HTTPException(status_code=422, detail="The uploaded video needs a filename")

        job_id = str(uuid.uuid4())
        suffix = Path(video.filename).suffix
        if not SAFE_SUFFIX.fullmatch(suffix):
            suffix = ".video"
        upload_directory = manager.upload_root / job_id
        upload_directory.mkdir(parents=True, exist_ok=False)
        upload_path = upload_directory / f"input{suffix.lower()}"
        size = 0
        try:
            with upload_path.open("wb") as stream:
                while chunk := await video.read(1024 * 1024):
                    size += len(chunk)
                    if size > app.state.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Uploaded video is too large")
                    stream.write(chunk)
        except Exception:
            upload_path.unlink(missing_ok=True)
            shutil.rmtree(upload_directory, ignore_errors=True)
            raise
        finally:
            await video.close()
        if size == 0:
            upload_path.unlink(missing_ok=True)
            upload_directory.rmdir()
            raise HTTPException(status_code=422, detail="Uploaded video is empty")

        options = {"objects": {"detector_backend": "omdet_turbo"}}
        object_dependent = {"objects", "hand_object_interactions", "vlm_descriptions"}
        if object_dependent.intersection(requested) and object_concepts:
            concepts = [
                " ".join(value.strip().split())
                for value in object_concepts.split(",")
                if value.strip()
            ]
            if len(concepts) > 40 or any(len(value) > 80 for value in concepts):
                upload_path.unlink(missing_ok=True)
                upload_directory.rmdir()
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Use at most 40 comma-separated object concepts, "
                        "each under 80 characters"
                    ),
                )
            if concepts:
                options["objects"]["concepts"] = concepts
        job = manager.create(video.filename, upload_path, requested, options)
        return _job_payload(job)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        try:
            return _job_payload(manager.get(job_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Job not found") from error

    @app.get("/api/jobs/{job_id}/overlay")
    def get_overlay(job_id: str) -> FileResponse:
        try:
            job = manager.get(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Job not found") from error
        if job.status != "completed":
            raise HTTPException(status_code=409, detail="Overlay is not ready")
        overlay = manager.session_root / job_id / "review" / "overlay.mp4"
        if not overlay.is_file():
            raise HTTPException(status_code=404, detail="Overlay was not produced")
        return FileResponse(overlay, media_type="video/mp4", filename=f"{job_id}-overlay.mp4")

    @app.get("/api/jobs/{job_id}/download")
    def download_dataset(job_id: str) -> FileResponse:
        try:
            job = manager.get(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Job not found") from error
        if job.status != "completed":
            raise HTTPException(status_code=409, detail="Dataset is not ready")
        archive = manager.download_root / f"{job_id}.zip"
        if not archive.is_file():
            raise HTTPException(status_code=404, detail="Dataset archive was not produced")
        return FileResponse(
            archive,
            media_type="application/zip",
            filename=f"mobocapture-{job_id}.zip",
        )

    @app.get("/api/jobs/{job_id}/redacted")
    def get_redacted(job_id: str) -> FileResponse:
        try:
            job = manager.get(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Job not found") from error
        if job.status != "completed":
            raise HTTPException(status_code=409, detail="Redacted video is not ready")
        redacted = manager.session_root / job_id / "review" / "redacted.mp4"
        if not redacted.is_file():
            raise HTTPException(status_code=404, detail="Redacted video was not produced")
        return FileResponse(
            redacted, media_type="video/mp4", filename=f"{job_id}-redacted.mp4"
        )

    return app
