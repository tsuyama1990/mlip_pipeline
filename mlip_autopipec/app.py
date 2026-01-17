"""
Main CLI application for MLIP-AutoPipe.
"""
import logging
import webbrowser
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.logging import RichHandler

from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.core.logging import setup_logging

# Configure basic logging for app startup
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

app = typer.Typer(help="MLIP-AutoPipe: Zero-Human Machine Learning Interatomic Potentials")
console = Console()
log = logging.getLogger(__name__)

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
    """Execute the MLIP-AutoPipe workflow."""
    try:
        console.print(f"[bold blue]MLIP-AutoPipe[/bold blue]: Launching run for {config_file.name}")

        # 1. Config Factory: Load and Expand
        config = ConfigFactory.from_yaml(config_file)

        # 2. Setup Logging
        setup_logging(config.log_path)
        log.info("System logging initialized.")

        # 3. Initialize Database
        db = DatabaseManager(config.db_path)
        db.initialize(config)
        log.info(f"System initialized. DB: {config.db_path}")

        console.print("[bold green]SUCCESS:[/bold green] System initialized successfully.")

    except FileNotFoundError as e:
        console.print(f"[bold red]FILE ERROR:[/bold red] {e}")
        logging.debug("Exception traceback:", exc_info=True)
        raise typer.Exit(code=1)
    except ValidationError as e:
        console.print(f"[bold red]CONFIGURATION ERROR:[/bold red] {e}")
        # The UAT says "user should not see a raw Python traceback", but e is formatted by Pydantic?
        # Maybe we want to print str(e) which is readable.
        logging.debug("Exception traceback:", exc_info=True)
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]FAILURE:[/bold red] An unexpected error occurred: {e}")
        logging.exception("Unhandled exception during workflow execution.")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
