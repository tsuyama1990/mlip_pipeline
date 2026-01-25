"""
Main CLI application for MLIP-AutoPipe.

This module provides the command-line interface using `typer`.
It acts as a facade, delegating business logic to handlers.
"""

import logging
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console

from mlip_autopipec.modules.cli_handlers.handlers import CLIHandler
from mlip_autopipec.utils.logging import setup_logging

app = typer.Typer(help="MLIP-AutoPipe: Zero-Human Machine Learning Interatomic Potentials")
db_app = typer.Typer(help="Database management commands")
run_app = typer.Typer(help="Execution commands")
app.add_typer(db_app, name="db")
app.add_typer(run_app, name="run")

console = Console()
logger = logging.getLogger("mlip_autopipec")


@run_app.callback(invoke_without_command=True)
def run_main(
    ctx: typer.Context,
    config_file: Path = typer.Option(  # noqa: B008
        Path("input.yaml"), "--config", "-c", help="Config file"
    ),
) -> None:
    """Execution commands. Default: Run the full workflow loop."""
    if ctx.invoked_subcommand is None:
        setup_logging()
        try:
            CLIHandler.run_loop(config_file)
        except Exception as e:
            console.print(f"[bold red]Workflow Failed:[/bold red] {e}")
            logger.exception("Workflow failed")
            raise typer.Exit(code=1) from e


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
    file: Path = typer.Argument(Path("input.yaml"), help="Path to config file"),
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


if __name__ == "__main__":
    app()
