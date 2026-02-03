import logging
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from mlip_autopipec.config.config_model import SimulationConfig
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.utils.logging import setup_logging

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.callback()
def callback() -> None:
    """MLIP Pipeline CLI."""


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to configuration file", exists=True),  # noqa: B008
) -> None:
    """Run the active learning pipeline."""
    setup_logging()

    try:
        with config_path.open() as f:
            config_data = yaml.safe_load(f)

        # Validate config
        config = SimulationConfig(**config_data)
        logger.info(f"PYACEMAKER initialized for project: {config.project_name}")
        logger.info("Configuration loaded successfully.")

        # Run orchestrator
        orchestrator = Orchestrator(config)
        orchestrator.run()

    except ValidationError as e:
        logger.error(f"Configuration validation failed:\n{e}")  # noqa: TRY400
        raise typer.Exit(code=1) from e
    except Exception as e:
        logger.error(f"An error occurred: {e}")  # noqa: TRY400
        raise typer.Exit(code=1) from e
