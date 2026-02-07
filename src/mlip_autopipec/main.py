import logging
from pathlib import Path

import typer
import yaml

from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.factory import (
    create_dynamics,
    create_generator,
    create_oracle,
    create_selector,
    create_trainer,
    create_validator,
)
from mlip_autopipec.utils.logging import setup_logging

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def init(output: Path = Path("config.yaml")) -> None:
    """
    Initialize a sample configuration file.
    """
    sample_config = {
        "workdir": "experiment_01",
        "max_cycles": 10,
        "oracle": {"type": "mock", "params": {}},
        "trainer": {"type": "mock", "params": {"dummy_file_name": "model.yace"}},
        "dynamics": {"type": "mock", "halt_probability": 0.5, "params": {}},
        "generator": {"type": "mock", "n_candidates": 5, "params": {}},
        "validator": {"type": "mock", "params": {}},
        "selector": {"type": "mock", "params": {}},
    }

    with output.open("w") as f:
        yaml.dump(sample_config, f, default_flow_style=False)

    logger.info(f"Sample configuration written to {output}")


@app.command()
def check_config(config_path: Path) -> None:
    """
    Validate a configuration file.
    """
    setup_logging()
    if not config_path.exists():
        logger.error(f"Config file {config_path} not found.")
        raise typer.Exit(code=1)

    try:
        with config_path.open() as f:
            data = yaml.safe_load(f)
        GlobalConfig.model_validate(data)
        logger.info("Configuration is valid.")
    except Exception as e:
        logger.exception("Configuration invalid")
        raise typer.Exit(code=1) from e


@app.command()
def run(config_path: Path) -> None:
    """
    Run the pipeline (Placeholder for Cycle 02).
    """
    setup_logging()

    if not config_path.exists():
        logger.error(f"Config file {config_path} not found.")
        raise typer.Exit(code=1)

    try:
        with config_path.open() as f:
            data = yaml.safe_load(f)
        config = GlobalConfig.model_validate(data)
    except Exception as e:
        logger.exception("Failed to load config")
        raise typer.Exit(code=1) from e

    try:
        # Factory test
        oracle = create_oracle(config.oracle)
        trainer = create_trainer(config.trainer)
        dynamics = create_dynamics(config.dynamics)
        generator = create_generator(config.generator)
        validator = create_validator(config.validator)
        selector = create_selector(config.selector)

        logger.info(
            f"Initialized components: {oracle}, {trainer}, {dynamics}, {generator}, {validator}, {selector}"
        )
        logger.info("All components initialized successfully.")

    except Exception as e:
        logger.exception("Failed to initialize components")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
