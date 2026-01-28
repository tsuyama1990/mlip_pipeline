import logging
from pathlib import Path
from typing import Annotated

import typer

from mlip_autopipec.config.loaders.yaml_loader import load_config
from mlip_autopipec.config.schemas.core import UserInputConfig
from mlip_autopipec.modules.cli_handlers.handlers import CLIHandler
from mlip_autopipec.orchestration.workflow import WorkflowManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("mlip_pipeline.log")],
)
logger = logging.getLogger("mlip_app")

app = typer.Typer(
    name="mlip-auto",
    help="Automated Machine Learning Interatomic Potential Creation (Cycle 02)",
    add_completion=False,
)


@app.command()
def run(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to the configuration YAML file.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    work_dir: Annotated[
        Path,
        typer.Option(
            "--work-dir",
            "-w",
            help="Working directory for artifacts.",
            file_okay=False,
            writable=True,
        ),
    ] = Path("workspace"),
    state: Annotated[
        Path | None,
        typer.Option("--state", "-s", help="Path to a workflow state file to resume from."),
    ] = None,
):
    """
    Starts the automated active learning cycle.
    """
    try:
        # Load Configuration
        logger.info(f"Loading configuration from {config}")
        user_config = load_config(config, UserInputConfig)

        # Initialize Workflow Manager
        manager = WorkflowManager(config=user_config, work_dir=work_dir, state_file=state)

        # Run Workflow
        logger.info("Starting Workflow...")
        manager.run()
        logger.info("Workflow Completed Successfully.")

    except Exception:
        logger.exception("Workflow failed.")
        raise typer.Exit(code=1)


@app.command()
def validate(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Path to configuration file", exists=True)
    ],
    phonon: bool = typer.Option(False, help="Run phonon validation"),
    elastic: bool = typer.Option(False, help="Run elasticity validation"),
    eos: bool = typer.Option(False, help="Run EOS validation"),
):
    """
    Runs physics validation (Phonon, Elastic, EOS).
    """
    try:
        CLIHandler.run_physics_validation(config, phonon, elastic, eos)
    except Exception:
        logger.exception("Validation failed")
        raise typer.Exit(code=1)


@app.command()
def run_dft(
    config: Annotated[Path, typer.Option(help="Path to config file")],
    structure: Annotated[Path, typer.Option(help="Path to structure file (.xyz, .cif)")],
):
    """
    Runs a single DFT calculation for a structure (Utility).
    """
    try:
        CLIHandler.run_dft_calc(config, structure)
    except Exception:
        logger.exception("DFT Calculation failed")
        raise typer.Exit(code=1)


@app.command()
def run_cycle_02(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Path to configuration YAML", exists=True)
    ],
):
    """
    Executes Cycle 02 Pipeline: Generation -> DFT -> DB -> Training (One-Shot).
    """
    try:
        CLIHandler.run_cycle_02(config)
    except Exception:
        logger.exception("Cycle 02 pipeline failed")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
