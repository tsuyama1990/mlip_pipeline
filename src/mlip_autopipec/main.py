import logging
from pathlib import Path
from typing import Annotated

import typer
import yaml

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.orchestration.mocks import (
    MockExplorer,
    MockOracle,
    MockTrainer,
    MockValidator,
)
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.utils.logging import setup_logging

app = typer.Typer(name="mlip-pipeline", help="MLIP Automated Pipeline CLI")
logger = logging.getLogger(__name__)


@app.callback()
def callback() -> None:
    """MLIP Pipeline CLI"""


@app.command()
def run(
    config: Annotated[Path, typer.Option(..., help="Path to configuration YAML file")],
) -> None:
    """Run the active learning pipeline."""
    setup_logging()

    if not config.exists():
        logger.error(f"Config file not found: {config}")
        raise typer.Exit(code=1)

    try:
        with config.open() as f:
            config_data = yaml.safe_load(f)

        # Parse into GlobalConfig
        global_config = GlobalConfig(**config_data)

    except Exception:
        logger.exception("Error parsing config")
        raise typer.Exit(code=1) from None

    logger.info("Initializing components...")

    # In Cycle 01, we just use Mocks.
    # In future cycles, we might select implementation based on config.
    explorer = MockExplorer()
    oracle = MockOracle()
    trainer = MockTrainer()
    validator = MockValidator()

    orchestrator = Orchestrator(
        config=global_config,
        explorer=explorer,
        oracle=oracle,
        trainer=trainer,
        validator=validator,
    )

    try:
        orchestrator.run_loop()
    except Exception:
        logger.exception("Pipeline failed")
        raise typer.Exit(code=1) from None


def main() -> None:
    app()


if __name__ == "__main__":
    main()
