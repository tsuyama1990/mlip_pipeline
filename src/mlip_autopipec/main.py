import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.factory import (
    create_dynamics,
    create_generator,
    create_oracle,
    create_selector,
    create_trainer,
    create_validator,
)
from mlip_autopipec.infrastructure.mocks import MockOrchestrator
from mlip_autopipec.utils.logging import setup_logging

app = typer.Typer(
    help="PYACEMAKER (MLIP Pipeline) - Automated Active Learning for Interatomic Potentials."
)

@app.callback()
def main() -> None:
    """
    PYACEMAKER (MLIP Pipeline).
    """

@app.command()
def run(
    config_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the configuration file (YAML).",
        ),
    ],
) -> None:
    """
    Execute the pipeline based on the provided configuration.
    """
    try:
        # Load configuration
        config = GlobalConfig.from_yaml(config_path)
    except ValidationError as e:
        sys.stderr.write(f"Configuration Validation Error:\n{e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error loading configuration: {e}\n")
        sys.exit(1)

    # Setup logging
    setup_logging(config.workdir)
    logger = logging.getLogger("mlip_autopipec.main")

    logger.info("Configuration loaded successfully.")

    try:
        # Instantiate components (just to verify factories work as expected by Cycle 01)
        generator = create_generator(config.generator)
        logger.info(f"Initialised {generator.__class__.__name__}")

        oracle = create_oracle(config.oracle)
        logger.info(f"Initialised {oracle.__class__.__name__}")

        trainer = create_trainer(config.trainer)
        logger.info(f"Initialised {trainer.__class__.__name__}")

        dynamics = create_dynamics(config.dynamics)
        logger.info(f"Initialised {dynamics.__class__.__name__}")

        validator = create_validator(config.validator)
        logger.info(f"Initialised {validator.__class__.__name__}")

        selector = create_selector(config.selector)
        logger.info(f"Initialised {selector.__class__.__name__}")

        # Instantiate Orchestrator
        orchestrator = MockOrchestrator()
        orchestrator.run()

    except NotImplementedError:
        logger.exception("Component instantiation failed")
        sys.exit(1)
    except Exception:
        logger.exception("Runtime error")
        sys.exit(1)

if __name__ == "__main__":
    app()
