# ruff: noqa: D101, T201
"""Main CLI application for MLIP-AutoPipe."""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import typer
import yaml
from pydantic import ValidationError

from mlip_autopipec.config_schemas import UserConfig
from mlip_autopipec.utils.config_utils import (
    generate_system_config_from_user_config,
)
from mlip_autopipec.workflow_manager import WorkflowManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = typer.Typer()


@app.command()
def run(
    checkpoint_path: Path = typer.Option(
        Path("mlip_autopipec_checkpoint.json"),
        "--checkpoint",
        "-cp",
        help="Path to the checkpoint file.",
    ),
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the user configuration YAML file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    db_manager: Any = None,
    dft_factory: Any = None,
    trainer: Any = None,
) -> None:
    """Run the full MLIP-AutoPipe active learning workflow."""
    logging.info(f"Loading user configuration from: {config_path}")
    try:
        with open(config_path) as f:
            user_config_data = yaml.safe_load(f)
        user_config = UserConfig(**user_config_data)
        config = generate_system_config_from_user_config(user_config)
    except ValidationError as e:
        logging.error(f"Configuration error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during setup: {e}")
        raise typer.Exit(code=2)

    # Use provided instances or create default mocks
    db_manager = db_manager or MagicMock()
    dft_factory = dft_factory or MagicMock()
    trainer = trainer or MagicMock()

    client = setup_dask_client(config)
    client = setup_dask_client(config)
    manager = WorkflowManager(
        config=config,
        checkpoint_path=checkpoint_path,
        db_manager=db_manager,
        dft_factory=dft_factory,
        trainer=trainer,
        client=client,
    )
    manager.run()


if __name__ == "__main__":
    app()
