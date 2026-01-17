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
from mlip_autopipec.monitoring.dashboard import generate_dashboard

# Configure logging to use Rich
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

app = typer.Typer(help="MLIP-AutoPipe: Zero-Human Machine Learning Interatomic Potentials")
console = Console()

@app.callback()
def main() -> None:
    """MLIP-AutoPipe CLI Entry Point."""

@app.command()
def init(
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
    """Initialize a new project from a configuration file."""
    try:
        console.print("[bold blue]MLIP-AutoPipe[/bold blue]: Initializing project...")

        # Load Config
        # ConfigFactory.from_yaml creates directories
        system_config = ConfigFactory.from_yaml(config_file)

        # Setup Logging (Redirects to file + rich console)
        setup_logging(system_config.log_path)
        logger = logging.getLogger("mlip_autopipec")
        logger.info(f"Initializing project: {system_config.user_input.project_name}")

        # Initialize Database
        db = DatabaseManager(system_config.db_path)
        db.initialize(system_config)

        logger.info("System initialized successfully.")
        console.print(f"[bold green]SUCCESS:[/bold green] Project initialized at {system_config.working_dir}")

    except ValidationError as e:
        console.print(f"[bold red]CONFIGURATION ERROR:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]FAILURE:[/bold red] An unexpected error occurred: {e}")
        logging.exception("Unhandled exception during initialization.")
        raise typer.Exit(code=1)

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
        from mlip_autopipec.services.pipeline import PipelineController
        PipelineController.execute(config_file)
        console.print("[bold green]SUCCESS:[/bold green] Workflow finished.")
    except Exception as e:
        console.print(f"[bold red]FAILURE:[/bold red] {e}")
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

    except Exception as e:
        console.print(f"[bold red]FAILURE:[/bold red] {e}")
        logging.exception("Unhandled exception during dashboard generation.")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
