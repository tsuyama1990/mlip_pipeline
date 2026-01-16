"""
Main CLI application for MLIP-AutoPipe.
"""
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.config.loaders.yaml_loader import ConfigLoader
from mlip_autopipec.workflow_manager import WorkflowManager

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
        # 1. Load User Config
        console.print(f"[bold blue]Loading configuration from:[/bold blue] {config_file}")
        user_config = ConfigLoader.load_user_config(config_file)
        console.print(f"[green]✅ Validated project:[/green] [bold]{user_config.project_name}[/bold]")

        # 2. Create System Config
        system_config = ConfigFactory.from_user_input(user_config)

        # 3. Initialize Workflow
        summary = (
            f"Project: {system_config.project_name}\n"
            f"UUID: {system_config.run_uuid}\n"
            f"Elements: {', '.join(system_config.target_system.elements)}\n"
            f"Goal: {user_config.simulation_goal.type}"
        )
        console.print(Panel(summary, title="Workflow Initialization", border_style="blue"))

        manager = WorkflowManager(system_config=system_config, work_dir=Path.cwd())

        # 4. Run
        console.print("[bold yellow]🚀 Starting Workflow...[/bold yellow]")
        manager.run()
        console.print("[bold green]🎉 Workflow completed successfully![/bold green]")

    except Exception as e:
        console.print(f"[bold red]ERROR:[/bold red] {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
