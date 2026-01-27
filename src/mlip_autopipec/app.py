"""
Main CLI application for MLIP-AutoPipe.

This module provides the command-line interface using `typer`.
It acts as a facade, delegating business logic to handlers.
"""

import logging
from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.modules.cli_handlers.handlers import CLIHandler
from mlip_autopipec.utils.logging import setup_logging

app = typer.Typer(help="MLIP-AutoPipe: Zero-Human Machine Learning Interatomic Potentials")
db_app = typer.Typer(help="Database management commands")
run_app = typer.Typer(help="Execution commands")
app.add_typer(db_app, name="db")
app.add_typer(run_app, name="run")

console = Console()
logger = logging.getLogger("mlip_autopipec")


@app.command()
def init() -> None:
    """Initialize a new project with a template configuration file."""
    try:
        CLIHandler.init_project()
    except Exception as e:
        console.print(f"[bold red]Init Failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command(name="validate")
def validate(
    file: Path = typer.Argument(Path("input.yaml"), help="Path to config file"),  # noqa: B008
    phonon: bool = typer.Option(False, "--phonon", help="Run phonon validation"),
    elastic: bool = typer.Option(False, "--elastic", help="Run elasticity validation"),
    eos: bool = typer.Option(False, "--eos", help="Run EOS validation"),
) -> None:
    """
    Validate configuration or run physics checks on the trained potential.
    Default: Validate config.
    With flags: Run specified physics validation (requires trained potential).
    """
    setup_logging()
    try:
        if phonon or elastic or eos:
            CLIHandler.run_physics_validation(file, phonon=phonon, elastic=elastic, eos=eos)
        else:
            CLIHandler.validate_config(file)
    except ValidationError as e:
        console.print("[bold red]Validation Error:[/bold red]")
        console.print(str(e))
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command()
def generate(
    config_file: Path = typer.Option(  # noqa: B008
        Path("input.yaml"), "--config", "-c", help="Config file"
    ),
    dry_run: bool = typer.Option(False, help="Dry run without saving to DB"),
) -> None:
    """Generate initial training structures."""
    setup_logging()
    try:
        CLIHandler.generate_structures(config_file, dry_run)
    except Exception as e:
        console.print(f"[bold red]Generation Failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command()
def select(
    config_file: Path = typer.Option(  # noqa: B008
        Path("input.yaml"), "--config", "-c", help="Config file"
    ),
    n_samples: int = typer.Option(None, "--n", help="Number of samples to select (overrides config)"),
    model_type: str = typer.Option(None, "--model", help="Model type (overrides config)"),
) -> None:
    """Select diverse candidates using a surrogate model."""
    setup_logging()
    try:
        CLIHandler.select_candidates(config_file, n_samples, model_type)
    except Exception as e:
        console.print(f"[bold red]Selection Failed:[/bold red] {e}")
        logger.exception("Selection failed")
        raise typer.Exit(code=1) from e


@app.command()
def train(
    config_file: Path = typer.Option(  # noqa: B008
        Path("input.yaml"), "--config", "-c", help="Config file"
    ),
    prepare_only: bool = typer.Option(False, "--prepare-only", help="Only prepare data, do not train"),
) -> None:
    """Train a potential using Pacemaker."""
    setup_logging()
    try:
        CLIHandler.train_potential(config_file, prepare_only)
    except Exception as e:
        console.print(f"[bold red]Training Failed:[/bold red] {e}")
        logger.exception("Training failed")
        raise typer.Exit(code=1) from e


@db_app.command(name="init")
def db_init(
    config_file: Path = typer.Option(  # noqa: B008
        Path("input.yaml"), "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Initialize the database based on the configuration."""
    setup_logging()
    try:
        CLIHandler.init_db(config_file)
    except Exception as e:
        console.print(f"[bold red]Database Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@run_app.command(name="loop")
def run_loop(
    config_file: Path = typer.Option(  # noqa: B008
        Path("input.yaml"), "--config", "-c", help="Config file"
    ),
) -> None:
    """Run the full autonomous loop (Generation -> DFT -> Training -> Inference)."""
    setup_logging()
    try:
        CLIHandler.run_loop(config_file)
    except Exception as e:
        console.print(f"[bold red]Workflow Failed:[/bold red] {e}")
        logger.exception("Workflow failed")
        raise typer.Exit(code=1) from e


def _load_dft_config(path: Path) -> tuple[DFTConfig, Path]:
    """Helper to load DFT config from full MLIP config or standalone DFT config."""
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        msg = "Config must be a dictionary"
        raise ValueError(msg)

    work_dir = Path("dft_work") # Default

    if "dft" in data:
        # It's likely an MLIPConfig
        dft_config = DFTConfig(**data["dft"])
        if "runtime" in data and "work_dir" in data["runtime"]:
             work_dir = Path(data["runtime"]["work_dir"])
    else:
        # Assume standalone
        dft_config = DFTConfig(**data)

    return dft_config, work_dir

def _handle_error(msg: str) -> None:
    """Helper to print error and exit, abstracting raise."""
    console.print(f"[bold red]Error:[/bold red] {msg}")
    raise typer.Exit(code=1)

@app.command(name="run-dft")
def run_dft(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to DFT configuration YAML"),  # noqa: B008
    structure_path: Path = typer.Option(..., "--structure", "-s", help="Path to structure file (e.g., .cif, .xyz)"),  # noqa: B008
    work_dir: Path = typer.Option(None, "--work-dir", "-w", help="Override working directory"), # noqa: B008
) -> None:
    """Run a DFT calculation on a single structure."""
    try:
        from ase import Atoms
        from mlip_autopipec.dft.runner import QERunner

        console.print(f"Loading config from {config_path}...")
        config, config_work_dir = _load_dft_config(config_path)

        # Determine actual work dir
        base_work_dir = work_dir if work_dir else config_work_dir

        console.print(f"Loading structure from {structure_path}...")
        from ase.io import read
        # type: ignore[no-untyped-call]
        atoms_read = read(structure_path)

        atoms = atoms_read[0] if isinstance(atoms_read, list) else atoms_read

        if not isinstance(atoms, Atoms):
            _handle_error("Invalid structure file.")

        # Ensure unique work dir for structure
        run_work_dir = base_work_dir / structure_path.stem
        runner = QERunner(config=config, work_dir=run_work_dir)

        console.print(f"Running DFT calculation in {run_work_dir}...")
        result = runner.run(atoms)

        if result.converged:
            console.print("[bold green]DFT Calculation Successful![/bold green]")
            console.print(f"Energy: {result.energy} eV")
            forces = result.forces
            max_f = max(max(abs(f) for f in force) for force in forces) if forces else 0.0
            console.print(f"Max Force Component: {max_f} eV/A")
            console.print(f"Output saved to {run_work_dir}")
        else:
            console.print("[bold red]DFT Calculation Failed.[/bold red]")
            _handle_error(f"Error: {result.error_message}")

    except Exception as e:
        console.print(f"[bold red]An error occurred:[/bold red] {e}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
