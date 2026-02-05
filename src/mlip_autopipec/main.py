import logging
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.utils.logging import setup_logging

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)

@app.callback()
def callback() -> None:
    """
    MLIP Pipeline CLI
    """

@app.command()
def run(
    config_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to the configuration YAML file."), # noqa: B008
) -> None:
    """
    Run the MLIP active learning pipeline.
    """
    setup_logging()

    try:
        logger.info(f"Loading configuration from {config_path}")
        with config_path.open("r") as f:
            config_dict = yaml.safe_load(f)

        config = GlobalConfig(**config_dict)

    except ValidationError as e:
        logger.error(f"Configuration Validation Error:\n{e}") # noqa: TRY400
        raise typer.Exit(code=1) from e
    except Exception as e:
        logger.error(f"Error loading configuration: {e}") # noqa: TRY400
        raise typer.Exit(code=1) from e

    if config.execution_mode == "mock":
        logger.info("Initializing MOCK components...")
        explorer = MockExplorer()
        oracle = MockOracle()
        trainer = MockTrainer()
        validator = MockValidator()
    else:
        logger.error(f"Execution mode '{config.execution_mode}' is not yet supported in Cycle 01.")
        raise typer.Exit(code=1)

    try:
        orchestrator = Orchestrator(
            config=config,
            explorer=explorer,
            oracle=oracle,
            trainer=trainer,
            validator=validator
        )
        orchestrator.run()
    except Exception as e:
        logger.exception("An unexpected error occurred during execution.")
        raise typer.Exit(code=1) from e

if __name__ == "__main__":
    app()
