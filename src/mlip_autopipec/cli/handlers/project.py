"""Handler for project initialization."""

import os
from pathlib import Path

import typer

from mlip_autopipec.infrastructure import io


def init_project(path: Path) -> None:
    """
    Logic for initializing a new project.
    """
    if path.exists():
        typer.secho(f"File {path} already exists.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Use defaults from environment if available, otherwise hardcode sane defaults for the template
    # Note: We duplicate defaults here for the template generation,
    # but the Config object is the source of truth for runtime.
    template = {
        "project_name": os.getenv("MLIP_DEFAULT_PROJECT_NAME", "mlip_project"),
        "potential": {
            "elements": os.getenv("MLIP_DEFAULT_ELEMENTS", "Si").split(","),
            "cutoff": float(os.getenv("MLIP_DEFAULT_CUTOFF", "5.0")),
            "seed": int(os.getenv("MLIP_DEFAULT_SEED", "42"))
        },
        "logging": {
            "level": os.getenv("MLIP_DEFAULT_LOG_LEVEL", "INFO"),
        }
    }

    try:
        io.dump_yaml(template, path)
        typer.secho(f"Created template configuration at {path}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Failed to create config: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
