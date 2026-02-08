import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models import GlobalConfig
from mlip_autopipec.utils import setup_logging

app = typer.Typer(name="mlip-pipeline", help="Active Learning Pipeline for MLIPs")
logger = logging.getLogger(__name__)


@app.callback()
def callback() -> None:
    """
    MLIP Pipeline CLI.
    """
    pass


@app.command()
def run(
    config_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to YAML configuration file",
        ),
    ],
) -> None:
    """
    Run the active learning pipeline using the specified configuration.
    """
    setup_logging()

    # Explicit security check
    if not config_path.is_file():
        logger.error(f"Configuration path {config_path} is not a valid file.")
        raise typer.Exit(code=2)

    try:
        with config_path.open("r") as f:
            config_data = yaml.safe_load(f)

        # Parse and validate config
        logger.info(f"Loading configuration from {config_path}")
        config = GlobalConfig(**config_data)

        # Run orchestrator
        orchestrator = Orchestrator(config)
        orchestrator.run()

    except ValidationError as e:
        logger.exception("Configuration validation error")
        raise typer.Exit(code=1) from e
    except Exception as e:
        logger.exception("Unexpected error")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
