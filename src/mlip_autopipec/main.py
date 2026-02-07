import logging
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from pydantic import ValidationError

from mlip_autopipec.domain_models import GlobalConfig
from mlip_autopipec.infrastructure import (
    MockDynamics,
    MockOracle,
    MockSelector,
    MockStructureGenerator,
    MockTrainer,
    MockValidator,
)
from mlip_autopipec.utils.logging import configure_logging

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def init(
    path: Annotated[
        Path, typer.Option("--path", "-p", help="Path to generate default config")
    ] = Path("config.yaml"),
) -> None:
    """
    Initialize a default configuration file.
    """
    default_config = {
        "workdir": "workdir",
        "max_cycles": 10,
        "oracle": {"type": "mock", "params": {}},
        "trainer": {"type": "mock", "params": {}},
        "dynamics": {"type": "mock", "params": {}},
        "generator": {"type": "mock", "params": {}},
        "validator": {"type": "mock", "params": {}},
        "selector": {"type": "mock", "params": {}},
    }
    with path.open("w") as f:
        yaml.dump(default_config, f)
    typer.echo(f"Generated default config at {path}")


@app.command()
def run(
    config_path: Annotated[Path, typer.Option("--config", "-c", help="Path to config file")] = Path(
        "config.yaml"
    ),
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Logging level")] = "INFO",
) -> None:
    """
    Run the MLIP pipeline.
    """
    configure_logging(log_level)

    if not config_path.exists():
        typer.echo(f"Config file {config_path} not found.")
        raise typer.Exit(code=1)

    try:
        with config_path.open() as f:
            config_dict = yaml.safe_load(f)
        config = GlobalConfig(**config_dict)
    except (ValidationError, yaml.YAMLError) as e:
        typer.echo(f"Invalid configuration: {e}")
        raise typer.Exit(code=1) from e

    logger.info(f"Loaded configuration from {config_path}")
    logger.info(f"Workdir: {config.workdir}")

    components: list[Any] = []

    # Factory instantiation
    if config.oracle.type == "mock":
        components.append(MockOracle(params=config.oracle.params))
        logger.info("Initialized MockOracle")

    if config.trainer.type == "mock":
        components.append(MockTrainer(params=config.trainer.params))
        logger.info("Initialized MockTrainer")

    if config.dynamics.type == "mock":
        components.append(MockDynamics(params=config.dynamics.params))
        logger.info("Initialized MockDynamics")

    if config.generator.type == "mock":
        components.append(MockStructureGenerator(params=config.generator.params))
        logger.info("Initialized MockStructureGenerator")

    if config.validator.type == "mock":
        components.append(MockValidator(params=config.validator.params))
        logger.info("Initialized MockValidator")

    if config.selector.type == "mock":
        components.append(MockSelector(params=config.selector.params))
        logger.info("Initialized MockSelector")

    logger.info("All components initialized successfully.")


@app.command()
def compute(
    structure_path: Annotated[Path, typer.Argument(help="Path to structure file (xyz/json)")],
    config_path: Annotated[Path, typer.Option("--config", "-c", help="Path to config file")] = Path(
        "config.yaml"
    ),
) -> None:
    """
    Run single-point calculation using the configured Oracle.
    """
    # Placeholder for Cycle 01
    typer.echo(f"Compute command placeholder for {structure_path}")


if __name__ == "__main__":
    app()
