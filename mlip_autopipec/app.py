"""
Main CLI application for MLIP-AutoPipe.
"""
import logging
import webbrowser
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from mlip_autopipec.core.bootstrap import initialize_project
from mlip_autopipec.exceptions import ConfigError, WorkspaceError, DatabaseError, LoggingError

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

        # Use bootstrap logic to separate concerns
        initialize_project(config_file)

        console.print("[bold green]SUCCESS:[/bold green] System initialized successfully.")

    except ConfigError as e:
        console.print(f"[bold red]CONFIGURATION ERROR:[/bold red] {e}")
        logging.debug("Exception traceback:", exc_info=True)
        raise typer.Exit(code=1)
    except WorkspaceError as e:
        console.print(f"[bold red]WORKSPACE ERROR:[/bold red] {e}")
        logging.debug("Exception traceback:", exc_info=True)
        raise typer.Exit(code=1)
    except DatabaseError as e:
        console.print(f"[bold red]DATABASE ERROR:[/bold red] {e}")
        logging.debug("Exception traceback:", exc_info=True)
        raise typer.Exit(code=1)
    except LoggingError as e:
        console.print(f"[bold red]LOGGING ERROR:[/bold red] {e}")
        logging.debug("Exception traceback:", exc_info=True)
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]FAILURE:[/bold red] An unexpected error occurred: {e}")
        logging.exception("Unhandled exception during workflow execution.")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
