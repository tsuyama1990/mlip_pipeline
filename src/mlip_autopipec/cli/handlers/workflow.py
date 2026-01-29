"""Handler for running the workflow."""

import logging
from pathlib import Path

import typer

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.infrastructure import logging as logging_infra
from mlip_autopipec.orchestration.workflow import WorkflowManager


def run_loop(config_path: Path) -> None:
    """
    Logic for running the active learning loop.
    """
    if not config_path.exists():
        typer.secho(f"Config file {config_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        # Load config
        config = Config.from_yaml(config_path)

        # Setup logging
        logging_infra.setup_logging(
            log_level=config.logging.level,
            log_file=config.logging.file_path
        )

        # Initialize and run workflow
        manager = WorkflowManager(config)
        manager.run()

    except Exception as e:
        typer.secho(f"Workflow failed: {e}", fg=typer.colors.RED)
        logging.exception("Workflow failed")
        raise typer.Exit(code=1) from e
