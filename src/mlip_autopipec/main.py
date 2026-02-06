import typer
import yaml
import logging
from pathlib import Path
from typing import Tuple

from mlip_autopipec.config import GlobalConfig
from mlip_autopipec.utils.logging import setup_logging
from mlip_autopipec.interfaces import BaseExplorer, BaseOracle, BaseTrainer, BaseValidator
from mlip_autopipec.infrastructure.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator
from mlip_autopipec.orchestration.orchestrator import Orchestrator

app = typer.Typer()

@app.callback()
def main() -> None:
    """
    MLIP Pipeline CLI
    """
    pass

def load_config(config_path: Path) -> GlobalConfig:
    if not config_path.exists():
        typer.echo(f"Error: Configuration file {config_path} not found.", err=True)
        raise typer.Exit(code=1)

    with config_path.open("r") as f:
        try:
            data = yaml.safe_load(f)
            return GlobalConfig(**data)
        except Exception as e:
            typer.echo(f"Error validating configuration: {e}", err=True)
            raise typer.Exit(code=1) from None

def get_components(config: GlobalConfig) -> Tuple[BaseExplorer, BaseOracle, BaseTrainer, BaseValidator]:
    # Factory logic (simplified for Cycle 01)
    explorer: BaseExplorer
    oracle: BaseOracle
    trainer: BaseTrainer
    validator: BaseValidator

    if config.explorer.type == "mock":
        explorer = MockExplorer()
    else:
        msg = f"Explorer type {config.explorer.type} not implemented"
        raise NotImplementedError(msg)

    if config.oracle.type == "mock":
        oracle = MockOracle()
    else:
        msg = f"Oracle type {config.oracle.type} not implemented"
        raise NotImplementedError(msg)

    if config.trainer.type == "mock":
        trainer = MockTrainer()
    else:
        msg = f"Trainer type {config.trainer.type} not implemented"
        raise NotImplementedError(msg)

    # Validator currently hardcoded to Mock as it wasn't in config explictly in SPEC 3.1
    validator = MockValidator()

    return explorer, oracle, trainer, validator

@app.command()
def run(config: Path = typer.Option(..., help="Path to the configuration YAML file.")) -> None:  # noqa: B008
    """
    Run the active learning pipeline.
    """
    # 1. Setup Logging
    setup_logging()

    # 2. Load Config
    global_config = load_config(config)

    # 3. Instantiate Components
    explorer, oracle, trainer, validator = get_components(global_config)

    # 4. Initialize Orchestrator
    orchestrator = Orchestrator(
        config=global_config,
        explorer=explorer,
        oracle=oracle,
        trainer=trainer,
        validator=validator
    )

    # 5. Run
    orchestrator.run()

if __name__ == "__main__":
    app()
