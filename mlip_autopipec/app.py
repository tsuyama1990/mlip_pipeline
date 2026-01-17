"""
Main CLI application for MLIP-AutoPipe.
"""
import logging
import webbrowser
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.logging import RichHandler

from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.config.models import UserInputConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.core.logging import setup_logging
from mlip_autopipec.monitoring.dashboard import generate_dashboard
from mlip_autopipec.services.pipeline import PipelineController

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
    """Initialize a new MLIP-AutoPipe project."""
    try:
        console.print(f"[bold blue]MLIP-AutoPipe[/bold blue]: Initializing project from {config_file.name}")

        # Load and validate user config
        with config_file.open() as f:
            user_config_dict = yaml.safe_load(f)
        user_config = UserInputConfig.model_validate(user_config_dict)

        # Create SystemConfig and directory
        system_config = ConfigFactory.from_user_input(user_config)
        # Note: ConfigFactory.from_user_input creates the directory as a side effect currently.
        # Ensure directory exists (ConfigFactory should have done it, but double check)
        if system_config.working_dir and not system_config.working_dir.exists():
             system_config.working_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"Created project directory at: {system_config.working_dir}")

        # Setup Logging inside the project
        # Only setup file logging if working_dir is set
        if system_config.working_dir:
            log_path = system_config.working_dir / "mlip_auto.log"
            setup_logging(log_path, level="INFO")
            logging.info(f"Initialized project: {system_config.project_name}")

            # Initialize Database
            db_path = system_config.working_dir / system_config.db_path
            db_manager = DatabaseManager(db_path)
            db_manager.initialize(system_config)
            logging.info(f"Initialized database at: {db_path}")

        console.print("[bold green]SUCCESS:[/bold green] Project initialized.")

    except (ValidationError, ValueError) as e:
        console.print(f"[bold red]CONFIGURATION ERROR:[/bold red] {e}")
        # logging might not be setup yet if it failed early, but basicConfig is there
        logging.debug("Exception traceback:", exc_info=True)
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
        PipelineController.execute(config_file)
        console.print("[bold green]SUCCESS:[/bold green] Workflow finished.")
    except FileNotFoundError as e:
        console.print(f"[bold red]FILE ERROR:[/bold red] {e}")
        logging.debug("Exception traceback:", exc_info=True)
        raise typer.Exit(code=1)
    except ValidationError as e:
        console.print(f"[bold red]CONFIGURATION ERROR:[/bold red] {e}")
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
