"""
Main CLI application for MLIP-AutoPipe.

This module serves as the entry point for the application, handling command-line
arguments and orchestrating the high-level workflow.
"""

from pathlib import Path

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.config.models import UserInputConfig
from mlip_autopipec.workflow_manager import WorkflowManager

# Initialize Typer and Rich Console
app = typer.Typer(help="MLIP-AutoPipe: Zero-Human Machine Learning Interatomic Potentials")
console = Console()


@app.callback()
def main() -> None:
    """
    MLIP-AutoPipe: Zero-Human Machine Learning Interatomic Potentials.
    """


class WorkflowError(Exception):
    """Custom exception for high-level workflow failures."""



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
    Execute the MLIP-AutoPipe workflow.

    This command parses the user's configuration file, initializes the system,
    and launches the automated workflow.
    """
    try:
        # Step 1: Load and Parse Config
        console.print(f"[bold blue]Loading configuration from:[/bold blue] {config_file}")

        try:
            with open(config_file) as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            console.print(f"[bold red]ERROR: Failed to parse YAML file:[/bold red]\n{e}")
            raise typer.Exit(code=1) from e

        try:
            user_config = UserInputConfig.model_validate(raw_config)
            console.print(
                f"[green]✅ Configuration validated for project:[/green] [bold]{user_config.project_name}[/bold]"
            )
        except ValidationError as e:
            console.print("[bold red]ERROR: Configuration validation failed:[/bold red]")
            console.print(str(e))
            raise typer.Exit(code=1) from e

        # Step 2: Initialize Workflow
        try:
            system_config = ConfigFactory.from_user_input(user_config)

            # Create a working directory if it doesn't exist?
            # The SPEC implies we run in the current directory or handle it internally.
            # WorkflowManager takes work_dir. We'll use the current working directory.
            work_dir = Path.cwd()

            # Display a nice summary panel
            summary = (
                f"Project: {system_config.project_name}\n"
                f"UUID: {system_config.run_uuid}\n"
                f"Elements: {', '.join(system_config.target_system.elements)}\n"
                f"Goal: {user_config.simulation_goal.type}"
            )
            console.print(Panel(summary, title="Workflow Initialization", border_style="blue"))

            manager = WorkflowManager(system_config=system_config, work_dir=work_dir)

        except Exception as e:
            console.print(f"[bold red]ERROR: Failed to initialize workflow:[/bold red]\n{e}")
            # We log the full exception to help debugging but show a clean message to user
            # logging.exception("Workflow initialization failed")
            raise typer.Exit(code=2) from e

        # Step 3: Launch Workflow
        console.print("[bold yellow]🚀 Starting Workflow...[/bold yellow]")
        try:
            manager.run()
            console.print("[bold green]🎉 Workflow completed successfully![/bold green]")
        except Exception as e:
            # Catching generic Exception here as WorkflowManager might raise various errors
            # ideally we would catch specific WorkflowErrors
            console.print(f"[bold red]ERROR: Workflow failed during execution:[/bold red]\n{e}")
            raise typer.Exit(code=3) from e

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[bold red]ERROR: An unexpected error occurred:[/bold red]\n{e}")
        raise typer.Exit(code=4) from e


if __name__ == "__main__":
    app()
