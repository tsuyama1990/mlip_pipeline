import logging
from pathlib import Path

import typer

from mlip_autopipec.constants import (
    DEFAULT_CUTOFF,
    DEFAULT_ELEMENTS,
    DEFAULT_LOG_FILENAME,
    DEFAULT_LOG_LEVEL,
    DEFAULT_PROJECT_NAME,
    DEFAULT_SEED,
)
from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.infrastructure import io
from mlip_autopipec.infrastructure import logging as logging_infra
from mlip_autopipec.orchestration import workflow


def init_project(path: Path) -> None:
    """
    Logic for initializing a new project.
    """
    if path.exists():
        typer.secho(f"File {path} already exists.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    template = {
        "project_name": DEFAULT_PROJECT_NAME,
        "potential": {
            "elements": DEFAULT_ELEMENTS,
            "cutoff": DEFAULT_CUTOFF,
            "seed": DEFAULT_SEED
        },
        "logging": {
            "level": DEFAULT_LOG_LEVEL,
            "file_path": DEFAULT_LOG_FILENAME
        },
        "lammps": {
            "command": "lmp_serial",
            "cores": 1,
            "timeout": 3600.0
        }
    }

    try:
        io.dump_yaml(template, path)
        typer.secho(f"Created template configuration at {path}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Failed to create config: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


def check_config(config_path: Path) -> None:
    """
    Logic for validating configuration.
    """
    if not config_path.exists():
        typer.secho(f"Config file {config_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        config = Config.from_yaml(config_path)
        logging_infra.setup_logging(config.logging)

        typer.secho("Configuration valid", fg=typer.colors.GREEN)
        logging.getLogger("mlip_autopipec").info("Validation successful")

    except Exception as e:
        typer.secho(f"Validation failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


def run_cycle_02(config_path: Path) -> None:
    """
    Logic for running Cycle 02 (One-Shot Pipeline).
    """
    if not config_path.exists():
        typer.secho(f"Config file {config_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        config = Config.from_yaml(config_path)
        logging_infra.setup_logging(config.logging)

        typer.secho("Starting Cycle 02...", fg=typer.colors.BLUE)
        result = workflow.run_one_shot(config)

        if result.status.value == "COMPLETED":
            typer.secho(f"Simulation Completed: Status {result.status.value}", fg=typer.colors.GREEN)
            typer.secho(f"Result in: {result.work_dir}", fg=typer.colors.GREEN)
        else:
            typer.secho(f"Simulation Failed: Status {result.status.value}", fg=typer.colors.RED)
            typer.secho(result.log_content, fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)

    except Exception as e:
        typer.secho(f"Execution failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
