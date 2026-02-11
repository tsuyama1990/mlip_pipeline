from pathlib import Path

import typer
from pydantic import ValidationError

from mlip_autopipec.core.config import load_config
from mlip_autopipec.core.orchestrator import Orchestrator

app = typer.Typer()

@app.callback()
def callback() -> None:
    """
    PyAceMaker (mlip-pipeline) CLI
    """

@app.command()
def run(config_path: Path) -> None:
    try:
        typer.echo(f"Loading configuration from {config_path}...")
        config = load_config(config_path)

        orchestrator = Orchestrator(config)
        orchestrator.run()

    except ValidationError as e:
        typer.echo(f"Validation Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e

if __name__ == "__main__":
    app()
