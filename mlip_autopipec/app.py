"""
Main CLI application for MLIP-AutoPipe.
"""
import logging
from pathlib import Path

import typer
from rich.console import Console

from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.core.logging import setup_logging
from mlip_autopipec.core.workspace import WorkspaceManager
from mlip_autopipec.exceptions import ConfigError, DatabaseError, MLIPError, WorkspaceError

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
        # 1. Load Config (Pure)
        config = ConfigFactory.from_yaml(input_file)

        # 2. Setup Workspace (Side Effects)
        workspace = WorkspaceManager(config)
        workspace.setup_workspace()

        # 3. Setup Logging
        setup_logging(config.log_path)

        # 4. Initialize Database
        db = DatabaseManager(config.db_path)
        db.initialize()
        db.set_system_config(config)

        log.info("System initialized")
        console.print("[bold green]System initialized successfully[/bold green]")

    except FileNotFoundError as e:
        console.print(f"[bold red]FILE ERROR:[/bold red] {e}")
        raise typer.Exit(code=1) from e
    except ConfigError as e:
        console.print(f"[bold red]CONFIGURATION ERROR:[/bold red] {e}")
        raise typer.Exit(code=1) from e
    except WorkspaceError as e:
        console.print(f"[bold red]WORKSPACE ERROR:[/bold red] {e}")
        raise typer.Exit(code=1) from e
    except DatabaseError as e:
        console.print(f"[bold red]DATABASE ERROR:[/bold red] {e}")
        raise typer.Exit(code=1) from e
    except MLIPError as e:
        console.print(f"[bold red]ERROR:[/bold red] {e}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[bold red]FAILURE:[/bold red] An unexpected error occurred: {e}")
        raise typer.Exit(code=1) from e

if __name__ == "__main__":
    app()
