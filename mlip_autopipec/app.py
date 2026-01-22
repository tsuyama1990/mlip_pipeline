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
from mlip_autopipec.generator import StructureBuilder

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


@app.command()
def generate(
    config_file: Path = typer.Option(
        Path("input.yaml"), "--config", "-c", help="Config file"
    ),
    dry_run: bool = typer.Option(False, help="Dry run without saving to DB"),
) -> None:
    """
    Generate initial training structures.
    """
    setup_logging()
    try:
        config = load_config(config_file)
        builder = StructureBuilder(config)
        structures = builder.build_batch()

        console.print(f"[green]Generated {len(structures)} structures.[/green]")

        if dry_run:
            console.print("[yellow]Dry run: Not saving to database.[/yellow]")
            return

        from mlip_autopipec.surrogate.candidate_manager import CandidateManager

        with DatabaseManager(config.runtime.database_path) as db:
            cm = CandidateManager(db)
            for atoms in structures:
                # Extract metadata
                metadata = atoms.info.copy()
                # CandidateManager handles defaults for status, generation etc.
                cm.create_candidate(atoms, metadata)

        console.print(f"[green]Saved to {config.runtime.database_path}[/green]")

    except Exception as e:
        console.print(f"[bold red]Generation Failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def select(
    config_file: Path = typer.Option(
        Path("input.yaml"), "--config", "-c", help="Config file"
    ),
    n_samples: int = typer.Option(None, "--n", help="Number of samples to select (overrides config)"),
    model_type: str = typer.Option(None, "--model", help="Model type (overrides config)"),
) -> None:
    """
    Select diverse candidates using a surrogate model.
    """
    setup_logging()
    try:
        from mlip_autopipec.surrogate.pipeline import SurrogatePipeline

        config = load_config(config_file)

        # Prepare Surrogate Config
        surrogate_conf = config.surrogate_config

        # Override with CLI args
        if n_samples is not None:
            surrogate_conf.n_samples = n_samples
        if model_type is not None:
            surrogate_conf.model_type = model_type

        with DatabaseManager(config.runtime.database_path) as db:
            pipeline = SurrogatePipeline(db, surrogate_conf)
            pipeline.run()

        console.print("[green]Selection complete.[/green]")

    except Exception as e:
        console.print(f"[bold red]Selection Failed:[/bold red] {e}")
        # Print full traceback for debug in dev
        import traceback
        traceback.print_exc()
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
