from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from mlip_autopipec.config.config_model import SimulationConfig
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.utils.logging import get_logger, setup_logging

app = typer.Typer(help="PYACEMAKER: Automated MLIP Pipeline")

# Placeholder callback to prevent Typer from collapsing into a single command
@app.callback()
def callback() -> None:
    """
    PYACEMAKER: Automated MLIP Pipeline
    """


@app.command()
def run(config_path: Path = typer.Argument(..., help="Path to the configuration YAML file"),  # noqa: B008
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")) -> None:
    """
    Run the automated pipeline with the given configuration.
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(name="mlip_autopipec", level=log_level)
    logger = get_logger("mlip_autopipec.main")

    try:
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            raise typer.Exit(code=1)  # noqa: TRY301

        # Load Config
        with config_path.open("r") as f:
            config_data = yaml.safe_load(f)

        try:
            config = SimulationConfig(**config_data)
        except ValidationError as e:
            logger.error("Configuration validation failed:")  # noqa: TRY400
            # Simplify error message for user
            for error in e.errors():
                 logger.error(f"  Field '{error['loc'][0]}' {error['msg']}")  # noqa: TRY400
            raise typer.Exit(code=1) from e

        # Initialize Components (Mocks for Cycle 01)
        # In future cycles, we would select implementation based on config.
        explorer = MockExplorer()
        oracle = MockOracle()
        trainer = MockTrainer()
        validator = MockValidator()

        # Initialize Orchestrator
        orchestrator = Orchestrator(
            config=config,
            explorer=explorer,
            oracle=oracle,
            trainer=trainer,
            validator=validator
        )

        # Run Loop
        orchestrator.run_loop()

    except typer.Exit:
        raise
    except Exception as e:
        logger.exception("An unexpected error occurred")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
