import logging
from pathlib import Path

import typer
import yaml

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.utils.logging import setup_logging

app = typer.Typer(pretty_exceptions_enable=False)
logger = logging.getLogger(__name__)

@app.callback()
def main_callback() -> None:
    """
    MLIP Pipeline CLI.
    """

@app.command()
def run(config_path: Path) -> None:
    """
    Run the active learning pipeline with the specified configuration.
    """
    try:
        if not config_path.exists():
            typer.echo(f"Error: Config file {config_path} not found.", err=True)
            raise typer.Exit(code=1) # noqa: TRY301

        with config_path.open("r") as f:
            config_data = yaml.safe_load(f)

        config = GlobalConfig(**config_data)

        # Setup logging
        # Ensure work_dir exists
        config.work_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(level=config.logging_level, log_file=str(config.work_dir / "mlip.log"))

        logger.info(f"Loaded configuration from {config_path}")

        # Initialize Mocks (Cycle 01)
        explorer = MockExplorer(config)
        oracle = MockOracle(config)
        trainer = MockTrainer(config)
        validator = MockValidator(config)

        # Run Orchestrator
        orch = Orchestrator(config, explorer, oracle, trainer, validator)
        orch.run()

        typer.echo("Pipeline completed successfully.")

    except Exception as e:
        typer.echo(f"Pipeline failed: {e}", err=True)
        # Only log if logging is set up, otherwise print to stderr
        # But we try to set up logging early.
        logger.exception("Pipeline failed")
        raise typer.Exit(code=1) from e

if __name__ == "__main__":
    app()
