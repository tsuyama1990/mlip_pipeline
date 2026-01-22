"""
Main CLI application for MLIP-AutoPipe.

This module provides the command-line interface using `typer`.
It supports:
- Project initialization (`init`)
- Configuration validation (`check-config`)
- Individual phase execution (`generate`, `select`, `train`)
- Full loop execution (`run loop`)
- Database management (`db init`)
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
run_app = typer.Typer(help="Execution commands")
app.add_typer(db_app, name="db")
app.add_typer(run_app, name="run")

console = Console()
logger = logging.getLogger("mlip_autopipec")


@app.command()
def init() -> None:
    """Initialize a new project with a template configuration file."""
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
            "command": "mpirun -np 4 pw.x",
            "pseudopotential_dir": "/path/to/upf",
            "ecutwfc": 40.0,
            "kspacing": 0.15,
            "nspin": 2,
        },
        "runtime": {"database_path": "mlip.db", "work_dir": "_work"},
        "training_config": {
             "cutoff": 5.0,
             "b_basis_size": 300,
             "kappa": 0.5,
             "kappa_f": 100.0,
             "max_iter": 100
        },
        "inference_config": {
            "lammps_executable": "/path/to/lmp",
            "temperature": 1000.0,
            "steps": 10000,
            "uncertainty_threshold": 10.0
        },
        "workflow": {
            "max_generations": 5,
            "workers": 4
        }
    }

    with open(input_file, "w") as f:
        yaml.dump(template, f, sort_keys=False)

    console.print("[green]Initialized new project. Please edit input.yaml.[/green]")


@app.command(name="check-config")
def check_config(file: Path = typer.Argument(..., help="Path to config file")) -> None:
    """Validate the configuration file."""
    setup_logging()
    try:
        load_config(file)
        console.print("[green]Validation Successful: Configuration is valid.[/green]")
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
    dry_run: bool = typer.Option(False, help="Dry run without saving to DB"),  # noqa: B008
) -> None:
    """Generate initial training structures."""
    setup_logging()
    try:
        config = load_config(config_file)
        # Use StructureBuilder (Module A)
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
        raise typer.Exit(code=1) from e


@app.command()
def select(
    config_file: Path = typer.Option(  # noqa: B008
        Path("input.yaml"), "--config", "-c", help="Config file"
    ),
    n_samples: int = typer.Option(None, "--n", help="Number of samples to select (overrides config)"),  # noqa: B008
    model_type: str = typer.Option(None, "--model", help="Model type (overrides config)"),  # noqa: B008
) -> None:
    """Select diverse candidates using a surrogate model."""
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
        logger.exception("Selection failed")
        raise typer.Exit(code=1) from e


@app.command()
def train(
    config_file: Path = typer.Option(  # noqa: B008
        Path("input.yaml"), "--config", "-c", help="Config file"
    ),
    prepare_only: bool = typer.Option(False, "--prepare-only", help="Only prepare data, do not train"),  # noqa: B008
) -> None:
    """Train a potential using Pacemaker."""
    setup_logging()
    try:
        # Use TrainingManager (Module D)
        from mlip_autopipec.modules.training_orchestrator import TrainingManager

        config = load_config(config_file)
        train_conf = config.training_config

        if not train_conf:
            console.print("[red]No training configuration found in input.yaml[/red]")
            raise typer.Exit(code=1)

        work_dir = config.runtime.work_dir
        db_path = config.runtime.database_path

        with DatabaseManager(db_path) as db:
            manager = TrainingManager(db, train_conf, work_dir)

            if prepare_only:
                from mlip_autopipec.training.dataset import DatasetBuilder
                builder = DatasetBuilder(db)
                builder.export(train_conf, work_dir)

                console.print(f"[green]Data preparation complete in {work_dir}[/green]")
                return

            result = manager.run_training()

            if result.success:
                console.print("[green]Training successful![/green]")
                if result.metrics:
                    console.print(f"Metrics: {result.metrics}")
                if result.potential_path:
                    console.print(f"Potential saved to: {result.potential_path}")
            else:
                console.print("[bold red]Training failed.[/bold red]")
                raise typer.Exit(code=1)

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
        config = load_config(config_file)
        db_manager = DatabaseManager(config.runtime.database_path)
        # ASE DB handles initialization on first connection/write
        with db_manager:
            pass # Just connect to initialize
        console.print(f"[green]Database initialized at {config.runtime.database_path}[/green]")
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
        from mlip_autopipec.config.models import SystemConfig
        from mlip_autopipec.orchestration.models import OrchestratorConfig
        from mlip_autopipec.orchestration.workflow import WorkflowManager

        config = load_config(config_file)

        # Extract Orchestrator Config from 'workflow_config'
        # If missing, provide defaults or raise error
        wf_config = config.workflow_config if config.workflow_config else OrchestratorConfig()

        manager = WorkflowManager(config, wf_config)
        manager.run()

        console.print("[green]Workflow finished.[/green]")

    except Exception as e:
        console.print(f"[bold red]Workflow Failed:[/bold red] {e}")
        logger.exception("Workflow failed")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
