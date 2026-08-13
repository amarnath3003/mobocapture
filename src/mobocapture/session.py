from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from mobocapture.io import sha256_file, write_json
from mobocapture.models import InputAsset, SessionManifest


class SessionWorkspace:
    def __init__(
        self,
        root: Path,
        raw_video: Path,
        manifest: SessionManifest,
        options: dict | None = None,
    ):
        self.root = root
        self.raw_video = raw_video
        self.manifest = manifest
        self.options = options or {}
        # Ephemeral, in-process state shared by processors in one run. Nothing
        # here is serialized into the dataset. It lets expensive model runtimes
        # be reused safely without creating process-global state between jobs.
        self.runtime: dict[str, Any] = {}

    @property
    def derived(self) -> Path:
        return self.root / "derived" / "v0.1"

    @property
    def review(self) -> Path:
        return self.root / "review"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @classmethod
    def create(
        cls,
        input_video: Path,
        root: Path,
        requested_modules: list[str],
        options: dict | None = None,
    ) -> "SessionWorkspace":
        input_video = input_video.resolve()
        root = root.resolve()
        if not input_video.is_file():
            raise FileNotFoundError(f"Input video does not exist: {input_video}")
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {root}")

        for relative in (
            "raw",
            "manifests",
            "processor_runs",
            "derived/v0.1",
            "review",
        ):
            (root / relative).mkdir(parents=True, exist_ok=True)

        source_hash = sha256_file(input_video)
        suffix = input_video.suffix.lower() or ".video"
        raw_video = root / "raw" / f"video{suffix}"
        temporary = raw_video.with_suffix(raw_video.suffix + ".copying")
        shutil.copy2(input_video, temporary)
        copied_hash = sha256_file(temporary)
        if copied_hash != source_hash:
            temporary.unlink(missing_ok=True)
            raise OSError("Copied video hash does not match the input")
        temporary.replace(raw_video)

        input_asset = InputAsset(
            original_name=input_video.name,
            stored_path=raw_video.relative_to(root).as_posix(),
            sha256=source_hash,
            size_bytes=raw_video.stat().st_size,
        )
        manifest = SessionManifest(
            session_id=str(uuid.uuid4()),
            status="processing",
            input=input_asset,
            requested_modules=requested_modules,
        )
        workspace = cls(root, raw_video, manifest, options)
        workspace.write_session_manifest()
        return workspace

    def write_session_manifest(self) -> None:
        write_json(self.manifests / "session.json", self.manifest)
