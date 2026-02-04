import logging
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.utils.logging import setup_logging

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.callback()
def main() -> None:
    """
    MLIP Pipeline CLI
    """


@app.command()
def run(config_path: Path) -> None:
    """
    Run the MLIP active learning pipeline using the provided configuration file.
    """
    setup_logging()

    if not config_path.exists():
        print(f"Error: Config file '{config_path}' not found.")  # noqa: T201
        raise typer.Exit(code=1)

    try:
        with config_path.open() as f:
            config_dict = yaml.safe_load(f)

        config = GlobalConfig(**config_dict)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")  # noqa: T201
        raise typer.Exit(code=1) from e
    except ValidationError as e:
        print(f"Config validation error: {e}")  # noqa: T201
        raise typer.Exit(code=1) from e
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")  # noqa: T201
        raise typer.Exit(code=1) from e

    logger.info("Configuration loaded successfully.")
    orch = Orchestrator(config)
    orch.run()
