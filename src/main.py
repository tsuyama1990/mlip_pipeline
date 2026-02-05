from pathlib import Path
from typing import Annotated

import typer
import yaml

from src.config.config_model import GlobalConfig
from src.orchestration.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator
from src.orchestration.orchestrator import Orchestrator
from src.utils.logging import setup_logging

app = typer.Typer()


@app.callback()
def main() -> None:
    """
    MLIP Pipeline CLI.
    """


@app.command()
def run(
    config_path: Annotated[Path, typer.Option("--config", help="Path to the configuration file")],
) -> None:
    """
    Run the MLIP active learning pipeline.
    """
    setup_logging()

    if not config_path.exists():
        typer.echo(f"Config file not found: {config_path}", err=True)
        raise typer.Exit(code=1)

    with config_path.open("r") as f:
        config_data = yaml.safe_load(f)

    try:
        config = GlobalConfig(**config_data)
    except Exception as e:
        typer.echo(f"Invalid configuration: {e}", err=True)
        raise typer.Exit(code=1) from e

    # Instantiate components (For Cycle 01, we use Mocks)
    explorer = MockExplorer()
    oracle = MockOracle()
    trainer = MockTrainer()
    validator = MockValidator()

    orchestrator = Orchestrator(
        config=config,
        explorer=explorer,
        oracle=oracle,
        trainer=trainer,
        validator=validator,
    )

    orchestrator.run()


if __name__ == "__main__":
    app()
