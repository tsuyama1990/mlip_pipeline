"""
Main CLI application for MLIP-AutoPipe.
"""

import logging
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console

from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.core.logging import setup_logging
from mlip_autopipec.core.services import load_config

app = typer.Typer(help="MLIP-AutoPipe: Zero-Human Machine Learning Interatomic Potentials")
db_app = typer.Typer(help="Database management commands")
app.add_typer(db_app, name="db")

console = Console()
logger = logging.getLogger("mlip_autopipec")


@app.command()
def init() -> None:
    """
    Initialize a new project with a template configuration file.
    """
    input_file = Path("input.yaml")
    if input_file.exists():
        console.print("[yellow]input.yaml already exists.[/yellow]")
        return

    # Template content
    template = {
        "target_system": {
            "elements": ["Fe", "Ni"],
            "composition": {"Fe": 0.7, "Ni": 0.3},
            "crystal_structure": "fcc",
        },
        "dft": {
            "pseudopotential_dir": "/path/to/upf",
            "ecutwfc": 40.0,
            "kspacing": 0.15,
            "nspin": 2,
        },
        "runtime": {"database_path": "mlip.db", "work_dir": "_work"},
    }

    with open(input_file, "w") as f:
        yaml.dump(template, f, sort_keys=False)

    console.print("[green]Initialized new project. Please edit input.yaml.[/green]")


@app.command(name="check-config")
def check_config(file: Path = typer.Argument(..., help="Path to config file")) -> None:
    """
    Validate the configuration file.
    """
    setup_logging()
    try:
        load_config(file)
        console.print("[green]Validation Successful: Configuration is valid.[/green]")
    except ValidationError as e:
        console.print("[bold red]Validation Error:[/bold red]")
        console.print(str(e))
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@db_app.command(name="init")
def db_init(
    config_file: Path = typer.Option(
        Path("input.yaml"), "--config", "-c", help="Path to config file"
    ),
) -> None:
    """
    Initialize the database based on the configuration.
    """
    setup_logging()
    try:
        config = load_config(config_file)
        db_manager = DatabaseManager(config.runtime.database_path)
        db_manager.initialize()
        console.print(f"[green]Database initialized at {config.runtime.database_path}[/green]")
    except Exception as e:
        console.print(f"[bold red]Database Error:[/bold red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
