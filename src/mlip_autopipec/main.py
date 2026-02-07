import logging
from pathlib import Path

import typer

from mlip_autopipec.config.base_config import GlobalConfig
from mlip_autopipec.orchestrator.simple_orchestrator import SimpleOrchestrator
from mlip_autopipec.utils.logging import configure_logging

app = typer.Typer()


@app.command()
def run(config_path: str = typer.Argument(..., help="Path to configuration YAML")) -> None:
    """
    Starts the Active Learning Orchestrator.
    """
    configure_logging()

    try:
        # Validate path
        path = Path(config_path)
        if not path.exists():
            typer.echo(f"Error: Configuration file {config_path} not found.", err=True)
            raise typer.Exit(code=1)  # noqa: TRY301

        config = GlobalConfig.from_yaml(config_path)
        orchestrator = SimpleOrchestrator(config)
        orchestrator.run()
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logging.getLogger(__name__).exception("Fatal error")
        raise typer.Exit(code=1) from e
