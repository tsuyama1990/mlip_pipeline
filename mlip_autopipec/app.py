"""
Main CLI application for MLIP-AutoPipe.
"""
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

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
    except Exception as e:
        console.print(f"[bold red]FAILURE:[/bold red] {e}")
        logging.debug("Exception traceback:", exc_info=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
