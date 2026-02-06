import logging
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from mlip_autopipec.config import GlobalConfig
from mlip_autopipec.infrastructure.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator
from mlip_autopipec.interfaces import BaseExplorer, BaseOracle, BaseTrainer, BaseValidator
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.utils.logging import setup_logging

app = typer.Typer()


@app.callback()
def main(
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."),
) -> None:
    """
    MLIP Pipeline CLI
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    setup_logging(level=level)


def load_config(config_path: Path) -> GlobalConfig:
    """
    Loads and validates the global configuration from a YAML file.
    """
    if not config_path.exists():
        typer.echo(f"Error: Configuration file {config_path} not found.", err=True)
        raise typer.Exit(code=1)

    with config_path.open("r") as f:
        try:
            data = yaml.safe_load(f)
            return GlobalConfig(**data)
        except yaml.YAMLError as e:
            typer.echo(f"Error parsing YAML configuration: {e}", err=True)
            raise typer.Exit(code=1) from None
        except ValidationError as e:
            typer.echo(f"Error validating configuration: {e}", err=True)
            raise typer.Exit(code=1) from None
        except Exception as e:
            typer.echo(f"Unexpected error loading configuration: {e}", err=True)
            raise typer.Exit(code=1) from None


def get_components(config: GlobalConfig) -> tuple[BaseExplorer, BaseOracle, BaseTrainer, BaseValidator]:
    """
    Instantiates the pipeline components (Explorer, Oracle, Trainer, Validator)
    based on the provided configuration.
    """
    explorer: BaseExplorer
    oracle: BaseOracle
    trainer: BaseTrainer
    validator: BaseValidator

    if config.explorer.type == "mock":
        explorer = MockExplorer(config.explorer, config.work_dir)
    else:
        msg = f"Explorer type {config.explorer.type} not implemented"
        raise NotImplementedError(msg)

    if config.oracle.type == "mock":
        oracle = MockOracle(config.work_dir)
    else:
        msg = f"Oracle type {config.oracle.type} not implemented"
        raise NotImplementedError(msg)

    if config.trainer.type == "mock":
        trainer = MockTrainer(config.trainer)
    else:
        msg = f"Trainer type {config.trainer.type} not implemented"
        raise NotImplementedError(msg)

    if config.validator.type == "mock":
        validator = MockValidator(config.validator)
    else:
        # Default mock if type matches or just simplified for now
        validator = MockValidator(config.validator)

    return explorer, oracle, trainer, validator


@app.command()
def run(config: Path = typer.Option(..., help="Path to the configuration YAML file.")) -> None:  # noqa: B008
    """
    Run the active learning pipeline.
    """
    # 1. Load Config
    global_config = load_config(config)

    # 2. Instantiate Components
    explorer, oracle, trainer, validator = get_components(global_config)

    # 3. Initialize Orchestrator
    orchestrator = Orchestrator(
        config=global_config,
        explorer=explorer,
        oracle=oracle,
        trainer=trainer,
        validator=validator,
    )

    # 4. Run
    orchestrator.run()


if __name__ == "__main__":
    app()
