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

# Import new Cycle 1 components
from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.core.logging import setup_logging
# Keep dashboard for status command if it works, or comment out if broken
from mlip_autopipec.monitoring.dashboard import generate_dashboard

# Configure basic logging for CLI before system logging is set up
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

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
    """Execute the MLIP-AutoPipe workflow."""
    try:
        console.print(f"[bold blue]MLIP-AutoPipe[/bold blue]: Launching run for {config_file.name}")

        # Step 1: Load and Validate Config
        config = ConfigFactory.from_yaml(config_file)

        # Step 2: Setup Logging
        setup_logging(config.log_path)
        # Re-get logger to ensure it uses the new config
        logger = logging.getLogger("mlip_autopipec")

        # Step 3: Initialize Database
        db = DatabaseManager(config.db_path)
        db.initialize(config)

        logger.info("System initialized successfully")
        console.print("[bold green]System initialized successfully[/bold green]")
        console.print(f"Working Directory: {config.working_dir}")
        console.print(f"Database: {config.db_path}")
        console.print(f"Log: {config.log_path}")

    except FileNotFoundError as e:
        console.print(f"[bold red]FILE ERROR:[/bold red] {e}")
        logging.debug("Exception traceback:", exc_info=True)
        raise typer.Exit(code=1)
    except ValidationError as e:
        console.print(f"[bold red]CONFIGURATION ERROR:[/bold red] {e}")
        # Make it human readable if possible, but raw error is also fine for now
        logging.debug("Exception traceback:", exc_info=True)
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]FAILURE:[/bold red] An unexpected error occurred: {e}")
        logging.exception("Unhandled exception during workflow execution.")
        raise typer.Exit(code=1)


@app.command()
def status(
    project_dir: Path = typer.Argument(
        ".",
        help="Path to the project directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the dashboard in a web browser."
    ),
) -> None:
    """Generate and view the project status dashboard."""
    console.print(f"[bold blue]MLIP-AutoPipe[/bold blue]: Generating dashboard for {project_dir}")
    try:
        dashboard_path = generate_dashboard(project_dir)
        console.print(f"[bold green]SUCCESS:[/bold green] Dashboard generated at {dashboard_path}")

        if open_browser:
            console.print("Opening dashboard in web browser...")
            webbrowser.open(f"file://{dashboard_path.absolute()}")

    except FileNotFoundError as e:
        console.print(f"[bold red]FILE ERROR:[/bold red] {e}")
        logging.debug("Exception traceback:", exc_info=True)
        raise typer.Exit(code=1)
    except RuntimeError as e:
        console.print(f"[bold red]RUNTIME ERROR:[/bold red] {e}")
        logging.debug("Exception traceback:", exc_info=True)
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]FAILURE:[/bold red] An unexpected error occurred: {e}")
        logging.exception("Unhandled exception during dashboard generation.")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
