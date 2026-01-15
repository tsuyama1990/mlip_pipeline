# ruff: noqa: D101, T201
"""Main CLI application for MLIP-AutoPipe."""

import logging
from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import ValidationError

from mlip_autopipec.config_schemas import UserConfig
from mlip_autopipec.data.database import DatabaseManager
from mlip_autopipec.modules.config_generator import PacemakerConfigGenerator
from mlip_autopipec.modules.dft.factory import DFTFactory
from mlip_autopipec.modules.trainer import PacemakerTrainer
from mlip_autopipec.utils.config_utils import (
    generate_system_config_from_user_config,
)
from mlip_autopipec.utils.workflow_utils import setup_dask_client
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

    # Use provided instances or create default implementations
    db_manager = db_manager or DatabaseManager(config.db_path)
    dft_factory = dft_factory or DFTFactory(config.dft)
    trainer = trainer or PacemakerTrainer(config, PacemakerConfigGenerator(config))

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
