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
from mlip_autopipec.orchestration.workflow import WorkflowManager


def init_project(path: Path) -> None:
    """
    Logic for initializing a new project.
    """
    if path.exists():
        typer.secho(f"File {path} already exists.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    template = {
        "project_name": DEFAULT_PROJECT_NAME,
        "logging": {
            "level": DEFAULT_LOG_LEVEL,
            "file_path": DEFAULT_LOG_FILENAME
        },
        "potential": {
            "elements": DEFAULT_ELEMENTS,
            "cutoff": DEFAULT_CUTOFF,
            "seed": DEFAULT_SEED
        },
        "structure_gen": {"enabled": False},
        "oracle": {"enabled": False},
        "trainer": {"enabled": False},
        "dynamics": {"enabled": False},
        "orchestrator": {}
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
        logging_infra.configure_logging(
            level=config.logging.level,
            log_file=config.logging.file_path
        )

        typer.secho("Configuration valid", fg=typer.colors.GREEN)
        logging.getLogger("mlip_autopipec").info("Validation successful")

    except Exception as e:
        typer.secho(f"Validation failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


def run_loop(config_path: Path) -> None:
    """
    Logic for running the active learning loop.
    """
    if not config_path.exists():
        typer.secho(f"Config file {config_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        config = Config.from_yaml(config_path)
        logging_infra.configure_logging(
            level=config.logging.level,
            log_file=config.logging.file_path
        )

        manager = WorkflowManager(config)
        manager.run()

    except Exception as e:
        typer.secho(f"Error running loop: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
