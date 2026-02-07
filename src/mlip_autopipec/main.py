import logging
from pathlib import Path
from typing import Annotated

import typer
import yaml

from mlip_autopipec.domain_models import GlobalConfig
from mlip_autopipec.orchestrator import SimpleOrchestrator
from mlip_autopipec.utils import configure_logging

app = typer.Typer()
logger = logging.getLogger(__name__)

@app.command()
def run(
    config: Annotated[Path, typer.Option(help="Path to configuration file")],
    log_level: Annotated[str, typer.Option(help="Logging level (DEBUG, INFO, WARNING, ERROR)")] = "INFO"
) -> None:
    """
    Run the active learning pipeline.
    """
    configure_logging(level=log_level)
    logger.info(f"Loading configuration from {config}")

    if not config.exists():
        logger.error(f"Configuration file {config} not found.")
        raise typer.Exit(code=1)

    try:
        with config.open("r") as f:
            data = yaml.safe_load(f)

        global_config = GlobalConfig(**data)

    except Exception:
        logger.exception("Failed to load configuration")
        raise typer.Exit(code=1) from None

    try:
        orchestrator = SimpleOrchestrator(global_config)
        orchestrator.run()
    except Exception:
        logger.exception("Orchestrator execution failed")
        raise typer.Exit(code=1) from None

@app.command()
def init(
    path: Annotated[Path, typer.Option(help="Path to save default configuration")] = Path("config.yaml")
) -> None:
    """
    Generate a default configuration file.
    """
    configure_logging()

    if path.exists():
        logger.warning(f"File {path} already exists. Aborting.")
        raise typer.Exit(code=1)

    default_config = {
        "project_name": "mlip_project_01",
        "seed": 42,
        "workdir": "mlip_run",
        "max_cycles": 5,
        "oracle": {"type": "mock", "params": {}},
        "trainer": {"type": "mock", "params": {}},
        "dynamics": {"type": "mock", "params": {}},
        "generator": {"type": "mock", "params": {}},
    }

    try:
        with path.open("w") as f:
            yaml.dump(default_config, f, sort_keys=False)
        logger.info(f"Default configuration created at {path}")
    except Exception:
        logger.exception("Failed to write configuration file")
        raise typer.Exit(code=1) from None

if __name__ == "__main__":
    app()
