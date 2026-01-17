"""
Main CLI application for MLIP-AutoPipe.
"""
import logging
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console

from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.core.logging import setup_logging

app = typer.Typer(help="MLIP-AutoPipe: Zero-Human Machine Learning Interatomic Potentials")
console = Console()
log = logging.getLogger(__name__)

@app.callback()
def main() -> None:
    """MLIP-AutoPipe CLI Entry Point."""

@app.command()
def run(
    input_file: Path = typer.Argument(  # noqa: B008
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to the input.yaml configuration file.",
    ),
) -> None:
    """Execute the MLIP-AutoPipe workflow."""
    try:
        config = ConfigFactory.from_yaml(input_file)
        setup_logging(config.log_path)

        db = DatabaseManager(config.db_path)
        db.initialize()
        db.set_system_config(config) # Store metadata

        log.info("System initialized")
        console.print("[bold green]System initialized successfully[/bold green]")

    except FileNotFoundError as e:
        console.print(f"[bold red]FILE ERROR:[/bold red] {e}")
        raise typer.Exit(code=1) from e
    except ValidationError as e:
        console.print(f"[bold red]CONFIGURATION ERROR:[/bold red] {e}")
        # Strip the technical stack trace for user, but log it?
        # The Spec says: "the error message should explicitly state: 'Composition must sum to 1.0'"
        # Pydantic errors contain this info.
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[bold red]FAILURE:[/bold red] An unexpected error occurred: {e}")
        raise typer.Exit(code=1) from e

if __name__ == "__main__":
    app()
