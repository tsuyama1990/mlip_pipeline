from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from mlip_autopipec.config.config_model import SimulationConfig
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.utils.logging import get_logger, setup_logging

app = typer.Typer()
logger = get_logger("mlip_autopipec.main")

@app.callback()
def callback() -> None:
    """
    PYACEMAKER CLI.
    """

@app.command()
def run(
    config_path: Path = typer.Argument(..., dir_okay=False, help="Path to configuration YAML file"), # noqa: B008
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    log_file: Path | None = typer.Option(None, "--log-file", help="Path to log file") # noqa: B008
) -> None:
    """
    Run the PYACEMAKER pipeline with the given configuration.
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level, log_file=str(log_file) if log_file else None)

    logger.info(f"Loading configuration from {config_path}")

    try:
        # Load and parse config
        with config_path.open() as f:
            config_dict = yaml.safe_load(f)

        config = SimulationConfig(**config_dict)
        logger.info("Configuration loaded successfully.")

        # Initialize Orchestrator
        orchestrator = Orchestrator(config)

        # Run
        orchestrator.run()

    except ValidationError as e:
        logger.error("Configuration validation failed:") # noqa: TRY400
        for err in e.errors():
            loc = " -> ".join(str(part) for part in err['loc'])
            logger.error(f"Field '{loc}': {err['msg']}") # noqa: TRY400
        raise typer.Exit(code=1) from e

    except Exception as e:
        logger.exception("An unexpected error occurred:")
        raise typer.Exit(code=1) from e

if __name__ == "__main__":
    app()
