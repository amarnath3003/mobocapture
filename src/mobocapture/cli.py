from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from mobocapture.modules import MODULES, PROFILES, resolve_modules, selected_modules
from mobocapture.pipeline import process_video


app = typer.Typer(
    name="mobocapture",
    help="Convert one RGB video into an evidence-linked MoboCapture session.",
    no_args_is_help=True,
)


@app.command("modules")
def list_modules() -> None:
    """List selectable RGB capability modules and implementation status."""
    for module in MODULES.values():
        resolved = resolve_modules([module.module_id])
        available = all(item.status == "ready" for item in resolved.processors)
        status = "ready" if available else "planned"
        typer.echo(f"{module.module_id:28} {status:8} {module.label} — {module.description}")
    typer.echo("\nProfiles: " + ", ".join(PROFILES))


@app.command("resolve")
def resolve_command(
    modules: Annotated[
        list[str] | None,
        typer.Option("--module", "-m", help="Capability module; repeat to select several."),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Named module preset."),
    ] = None,
) -> None:
    """Resolve module dependencies without processing a video."""
    try:
        requested = selected_modules(modules, profile)
        resolved = resolve_modules(requested)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(resolved.model_dump(mode="json"), indent=2))


@app.command("process")
def process_command(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", file_okay=False, help="New session directory."),
    ] = None,
    modules: Annotated[
        list[str] | None,
        typer.Option("--module", "-m", help="Capability module; repeat to select several."),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Named module preset."),
    ] = None,
    object_concepts: Annotated[
        list[str] | None,
        typer.Option(
            "--object-concept",
            help="Object name or phrase for open-vocabulary detection; repeat as needed.",
        ),
    ] = None,
    detector_backend: Annotated[
        str,
        typer.Option(
            "--detector-backend",
            help="Open-vocabulary detector: omdet_turbo (fast default) or grounding_dino.",
        ),
    ] = "omdet_turbo",
) -> None:
    """Process a single RGB video into a versioned local dataset."""
    output_directory = output or video.with_name(f"{video.stem}.mobocapture")
    try:
        workspace = process_video(
            input_video=video,
            output_directory=output_directory,
            module_ids=modules,
            profile=profile,
            options={
                "objects": {
                    **({"concepts": object_concepts} if object_concepts else {}),
                    "detector_backend": detector_backend,
                }
            },
        )
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(f"Session: {workspace.root}")
    typer.echo(f"Status:  {workspace.manifest.status}")
    typer.echo(f"Dataset: {workspace.derived}")
    typer.echo(f"Overlay: {workspace.review / 'overlay.mp4'}")
    if workspace.manifest.unavailable_processors:
        typer.echo(
            "Unavailable processors: " + ", ".join(workspace.manifest.unavailable_processors),
            err=True,
        )


@app.command("web")
def web_command(
    host: Annotated[
        str,
        typer.Option("--host", help="Interface to bind. Use 0.0.0.0 only on a trusted network."),
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
    data_directory: Annotated[
        Path,
        typer.Option("--data-dir", file_okay=False, help="Uploads, sessions, and ZIP outputs."),
    ] = Path("mobocapture-web-data"),
) -> None:
    """Launch the local upload and processing interface."""
    import uvicorn

    from mobocapture.web import create_app

    typer.echo(f"MoboCapture web UI: http://{host}:{port}")
    typer.echo(f"Data directory: {data_directory.resolve()}")
    uvicorn.run(create_app(data_directory), host=host, port=port)
