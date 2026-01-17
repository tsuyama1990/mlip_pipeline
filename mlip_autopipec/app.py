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


@app.callback()
def main() -> None:
    """MLIP-AutoPipe CLI Entry Point."""


@app.command()
def run(
    config_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to the input.yaml configuration file.",
    ),
) -> None:
    """
    Execute the MLIP-AutoPipe initialization (Cycle 01).
    Future cycles will extend this to run the full workflow.
    """
    try:
        console.print(
            f"[bold blue]MLIP-AutoPipe[/bold blue]: Initializing project from {config_file.name}"
        )

        # 1. Expand Config
        # Expand Config
        config = ConfigFactory.from_yaml(config_file)

        # 2. Setup Logging
        # Setup Logging
        setup_logging(config.log_path)
        log = logging.getLogger("mlip_autopipec")
        log.info(f"Logging initialized at {config.log_path}")

        # 3. Initialize Database
        # Initialize Database
        db_manager = DatabaseManager(config.db_path)
        db_manager.initialize(config)
        log.info(f"Database initialized at {config.db_path}")

        console.print(
            f"[bold green]SUCCESS:[/bold green] System initialized successfully. Working directory: {config.working_dir}"
        )
        log.info("System initialized successfully")

    except FileNotFoundError as e:
        console.print(f"[bold red]FILE ERROR:[/bold red] {e}")
        # logging might not be set up yet, so we use rich console
        raise typer.Exit(code=1) from e
    except ValidationError as e:
        console.print(f"[bold red]CONFIGURATION ERROR:[/bold red] {e}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[bold red]FAILURE:[/bold red] An unexpected error occurred: {e}")
        # If logging is setup, log it
        if logging.getLogger().handlers:
            logging.exception("Unhandled exception during execution.")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
