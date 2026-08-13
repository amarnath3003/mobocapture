from __future__ import annotations

import traceback
import uuid
from time import perf_counter
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mobocapture import __version__
from mobocapture.io import sha256_file, write_json
from mobocapture.models import ProcessorOutput, ProcessorRunManifest, utc_now
from mobocapture.session import SessionWorkspace


@dataclass
class ProcessorResult:
    outputs: list[Path]
    metrics: dict[str, Any] = field(default_factory=dict)


class Processor(ABC):
    processor_id: str
    version: str = __version__

    @abstractmethod
    def process(self, workspace: SessionWorkspace) -> ProcessorResult:
        raise NotImplementedError

    def run(self, workspace: SessionWorkspace) -> ProcessorRunManifest:
        run_id = str(uuid.uuid4())
        run_directory = workspace.root / "processor_runs" / self.processor_id / run_id
        manifest_path = run_directory / "manifest.json"
        manifest = ProcessorRunManifest(
            processor_id=self.processor_id,
            processor_version=self.version,
            run_id=run_id,
            status="running",
            started_at_utc=utc_now(),
            input_sha256=workspace.manifest.input.sha256,
        )
        write_json(manifest_path, manifest)
        try:
            started = perf_counter()
            result = self.process(workspace)
            result.metrics.setdefault(
                "wall_time_seconds", round(perf_counter() - started, 6)
            )
            outputs = []
            for output in result.outputs:
                outputs.append(
                    ProcessorOutput(
                        path=output.relative_to(workspace.root).as_posix(),
                        sha256=sha256_file(output),
                        size_bytes=output.stat().st_size,
                    )
                )
            manifest.status = "complete"
            manifest.completed_at_utc = utc_now()
            manifest.outputs = outputs
            manifest.metrics = result.metrics
            write_json(manifest_path, manifest)
            return manifest
        except Exception as error:
            manifest.status = "failed"
            manifest.completed_at_utc = utc_now()
            manifest.error = "".join(
                traceback.format_exception_only(type(error), error)
            ).strip()
            write_json(manifest_path, manifest)
            raise
