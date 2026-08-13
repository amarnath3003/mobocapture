from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable

from mobocapture.io import write_json, write_yaml
from mobocapture.models import RequestedModulesManifest, RunReport
from mobocapture.modules import resolve_modules, selected_modules
from mobocapture.processors import (
    HandTrackerProcessor,
    InteractionInferenceProcessor,
    ObjectDetectorProcessor,
    ObjectSegmenterProcessor,
    OpticalFlowProcessor,
    OverlayRendererProcessor,
    PointTrackerProcessor,
    PrivacyRedactorProcessor,
    PrivacyScannerProcessor,
    VideoIngestProcessor,
    VideoQualityProcessor,
)
from mobocapture.processors.base import Processor
from mobocapture.session import SessionWorkspace


IMPLEMENTED_PROCESSORS: dict[str, type[Processor]] = {
    "video_ingest": VideoIngestProcessor,
    "video_quality": VideoQualityProcessor,
    "hand_tracker": HandTrackerProcessor,
    "interaction_inference": InteractionInferenceProcessor,
    "object_detector": ObjectDetectorProcessor,
    "object_segmenter": ObjectSegmenterProcessor,
    "point_tracker": PointTrackerProcessor,
    "optical_flow": OpticalFlowProcessor,
    "privacy_scanner": PrivacyScannerProcessor,
    "privacy_redactor": PrivacyRedactorProcessor,
    "overlay_renderer": OverlayRendererProcessor,
}

ProgressCallback = Callable[[str, int, int], None]

GPU_PROCESSORS = {"object_detector", "privacy_scanner", "object_segmenter"}


def _ordered_processors(processors):
    """Keep dependency order while placing the shared DINO consumers together."""

    ordered = list(processors)
    ids = [item.processor_id for item in ordered]
    if "object_detector" in ids and "privacy_scanner" in ids:
        privacy = ordered.pop(ids.index("privacy_scanner"))
        detector_index = next(
            index
            for index, item in enumerate(ordered)
            if item.processor_id == "object_detector"
        )
        ordered.insert(detector_index + 1, privacy)
    # The overlay consumes every optional output and is always the final task.
    ordered.sort(key=lambda item: item.processor_id == "overlay_renderer")
    return ordered


def _run_processors(
    workspace: SessionWorkspace,
    processors,
    progress_callback: ProgressCallback | None,
):
    """Execute the processor DAG, overlapping independent CPU and GPU work.

    Learned GPU processors remain serialized so the output is deterministic and
    8 GB cards do not overcommit memory. Independent CPU work can proceed while
    the GPU is occupied. Results are returned in canonical dependency order,
    independent of completion timing.
    """

    ordered = _ordered_processors(processors)
    order = {item.processor_id: index for index, item in enumerate(ordered)}
    definitions = {item.processor_id: item for item in ordered}
    planned = set(definitions)
    workspace.runtime["planned_processor_ids"] = frozenset(planned)
    configured_workers = int(
        workspace.options.get("performance", {}).get("processor_workers", 3)
    )
    max_workers = max(1, min(configured_workers, os.cpu_count() or 1))
    completed: set[str] = set()
    submitted: set[str] = set()
    results = {}
    active: dict[Future, str] = {}

    def submit_ready(executor: ThreadPoolExecutor) -> bool:
        made_progress = False
        active_gpu = any(
            processor_id in GPU_PROCESSORS for processor_id in active.values()
        )
        ready = []
        for processor_id, definition in definitions.items():
            if processor_id in submitted:
                continue
            dependencies = set(definition.requires) & planned
            if dependencies <= completed:
                ready.append(definition)
        ready.sort(key=lambda item: order[item.processor_id])
        for definition in ready:
            if len(active) >= max_workers:
                break
            processor_id = definition.processor_id
            is_gpu = processor_id in GPU_PROCESSORS
            if is_gpu and active_gpu:
                continue
            if processor_id == "overlay_renderer" and len(completed) != len(ordered) - 1:
                continue
            if progress_callback:
                progress_callback(processor_id, len(completed), len(ordered))
            future = executor.submit(IMPLEMENTED_PROCESSORS[processor_id]().run, workspace)
            active[future] = processor_id
            submitted.add(processor_id)
            active_gpu = active_gpu or is_gpu
            made_progress = True
        return made_progress

    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="mobocapture-processor"
    ) as executor:
        while len(completed) < len(ordered):
            submitted_now = submit_ready(executor)
            if not active:
                if submitted_now:
                    continue
                unresolved = sorted(planned - completed)
                raise RuntimeError(
                    "Processor dependency graph cannot make progress: "
                    + ", ".join(unresolved)
                )
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                processor_id = active.pop(future)
                results[processor_id] = future.result()
                completed.add(processor_id)
                if progress_callback:
                    progress_callback(processor_id, len(completed), len(ordered))

    workspace.runtime.pop("planned_processor_ids", None)
    return [results[item.processor_id] for item in ordered], [
        item.processor_id for item in ordered
    ]


def process_video(
    input_video: Path,
    output_directory: Path,
    module_ids: list[str] | None = None,
    profile: str | None = None,
    options: dict | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SessionWorkspace:
    requested = selected_modules(module_ids, profile)
    resolved = resolve_modules(requested)
    processing_options = options or {}
    workspace = SessionWorkspace.create(
        input_video, output_directory, requested, options=processing_options
    )

    requested_manifest = RequestedModulesManifest(
        profile=profile,
        modules=requested,
        options=processing_options,
    )
    write_yaml(workspace.manifests / "requested_modules.yaml", requested_manifest)
    write_yaml(workspace.manifests / "resolved_processors.yaml", resolved)

    license_report = {
        "schema_version": "0.1.0",
        "policy": "record_every_component; do not run unaudited model weights",
        "processors": [
            {
                "processor_id": processor.processor_id,
                "code_license": processor.code_license,
                "model_license": processor.model_license,
                "status": processor.status,
            }
            for processor in resolved.processors
        ],
    }
    write_json(workspace.manifests / "license_report.json", license_report)

    completed: list[str] = []
    processor_runs = []
    runnable_processors = [
        processor
        for processor in resolved.processors
        if processor.status == "ready" and processor.processor_id in IMPLEMENTED_PROCESSORS
    ]
    try:
        processor_runs, completed = _run_processors(
            workspace, runnable_processors, progress_callback
        )
    except Exception:
        workspace.runtime.clear()
        workspace.manifest.status = "failed"
        workspace.manifest.resolved_processors = [item.processor_id for item in resolved.processors]
        workspace.manifest.unavailable_processors = [
            item.processor_id for item in resolved.processors if item.status == "unavailable"
        ]
        workspace.write_session_manifest()
        raise

    unavailable = [item for item in resolved.processors if item.status == "unavailable"]
    final_status = "complete_with_unavailable" if unavailable else "complete"
    provenance = {
        "schema_version": "0.1.0",
        "input": {
            "path": workspace.manifest.input.stored_path,
            "sha256": workspace.manifest.input.sha256,
            "provenance": "measured",
        },
        "processor_runs": [
            {
                "processor_id": run.processor_id,
                "processor_version": run.processor_version,
                "run_id": run.run_id,
                "input_sha256": run.input_sha256,
                "outputs": [output.model_dump(mode="json") for output in run.outputs],
                "run_manifest": (
                    Path("processor_runs") / run.processor_id / run.run_id / "manifest.json"
                ).as_posix(),
            }
            for run in processor_runs
        ],
    }
    write_json(workspace.derived / "provenance.json", provenance)
    if progress_callback:
        progress_callback("finalizing", len(runnable_processors), len(runnable_processors))
    report = RunReport(
        session_id=workspace.manifest.session_id,
        status=final_status,
        completed_processors=completed,
        unavailable_processors=unavailable,
        warnings=[
            "The review overlay is constant-frame-rate; use frame_index.parquet for canonical timing."
        ],
    )
    write_json(workspace.manifests / "run_report.json", report)
    workspace.manifest.status = final_status
    workspace.manifest.resolved_processors = [item.processor_id for item in resolved.processors]
    workspace.manifest.unavailable_processors = [item.processor_id for item in unavailable]
    workspace.write_session_manifest()
    return workspace
