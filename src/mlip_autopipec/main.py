import logging
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.orchestration.mocks import (
    MockExplorer,
    MockOracle,
    MockTrainer,
    MockValidator,
)
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.utils.logging import setup_logging

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.callback()
def callback() -> None:
    """
    MLIP Pipeline CLI
    """


@app.command()
def run(
    config_path: Path = typer.Argument(  # noqa: B008
        ..., exists=True, dir_okay=False, help="Path to the configuration YAML file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """
    Run the active learning pipeline.
    """
    try:
        log_level = "DEBUG" if verbose else "INFO"
        setup_logging(log_level)

        with config_path.open("r") as f:
            config_dict = yaml.safe_load(f)

        # Pydantic validation
        config = GlobalConfig(**config_dict)

        # Ensure work directory exists
        config.work_dir.mkdir(parents=True, exist_ok=True)

        if config.execution_mode == "mock":
            explorer = MockExplorer(work_dir=config.work_dir)
            oracle = MockOracle(work_dir=config.work_dir)
            trainer = MockTrainer(work_dir=config.work_dir)
            validator = MockValidator(work_dir=config.work_dir)
        else:
            print(  # noqa: T201
                "Production mode not implemented yet, using mocks as fallback."
            )
            explorer = MockExplorer(work_dir=config.work_dir)
            oracle = MockOracle(work_dir=config.work_dir)
            trainer = MockTrainer(work_dir=config.work_dir)
            validator = MockValidator(work_dir=config.work_dir)

        orchestrator = Orchestrator(
            explorer=explorer,
            oracle=oracle,
            trainer=trainer,
            validator=validator,
            config=config,
        )

        orchestrator.run()
        print("Workflow completed successfully")  # noqa: T201

    except ValidationError as e:
        print(f"Configuration Error: {e}")  # noqa: T201
        raise typer.Exit(code=1) from e
    except Exception as e:
        print(f"Unexpected Error: {e}")  # noqa: T201
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
