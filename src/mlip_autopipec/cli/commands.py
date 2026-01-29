import logging
from pathlib import Path

import typer

from mlip_autopipec.constants import (
    DEFAULT_CUTOFF,
    DEFAULT_ELEMENTS,
    DEFAULT_LOG_FILENAME,
    DEFAULT_LOG_LEVEL,
    DEFAULT_PROJECT_NAME,
    DEFAULT_SEED,
)
from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowPhase
from mlip_autopipec.infrastructure import io
from mlip_autopipec.infrastructure import logging as logging_infra
from mlip_autopipec.orchestration.phases import (
    CalculationPhase,
    ExplorationPhase,
    SelectionPhase,
    TrainingPhase,
    ValidationPhase,
)
from mlip_autopipec.orchestration.workflow import WorkflowManager


def init_project(path: Path) -> None:
    """
    Logic for initializing a new project.
    """
    if path.exists():
        typer.secho(f"File {path} already exists.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    template = {
        "project_name": DEFAULT_PROJECT_NAME,
        "potential": {
            "elements": DEFAULT_ELEMENTS,
            "cutoff": DEFAULT_CUTOFF,
            "seed": DEFAULT_SEED
        },
        "logging": {
            "level": DEFAULT_LOG_LEVEL,
            "file_path": DEFAULT_LOG_FILENAME
        },
        "orchestrator": {},
        "exploration": {},
        "selection": {},
        "dft": {},
        "training": {},
        "validation": {}
    }

    try:
        io.dump_yaml(template, path)
        typer.secho(f"Created template configuration at {path}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Failed to create config: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

def check_config(config_path: Path) -> None:
    """
    Logic for validating configuration.
    """
    if not config_path.exists():
        typer.secho(f"Config file {config_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        config = Config.from_yaml(config_path)
        logging_infra.setup_logging(config.logging)

        typer.secho("Configuration valid", fg=typer.colors.GREEN)
        logging.getLogger("mlip_autopipec").info("Validation successful")

    except Exception as e:
        typer.secho(f"Validation failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

def run_loop(config_path: Path) -> None:
    """
    Logic for running the Active Learning loop.
    """
    if not config_path.exists():
        typer.secho(f"Config file {config_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    logger = logging.getLogger("mlip_autopipec")

    try:
        # Load Config
        config = Config.from_yaml(config_path)
        logging_infra.setup_logging(config.logging)

        # Initialize Workflow Manager
        state_path = Path("workflow_state.json")
        manager = WorkflowManager(config, state_path)

        # Register Phases
        manager.register_phase(ExplorationPhase(), WorkflowPhase.EXPLORATION)
        manager.register_phase(SelectionPhase(), WorkflowPhase.SELECTION)
        manager.register_phase(CalculationPhase(), WorkflowPhase.CALCULATION)
        manager.register_phase(TrainingPhase(), WorkflowPhase.TRAINING)
        manager.register_phase(ValidationPhase(), WorkflowPhase.VALIDATION)

        # Run Cycle
        typer.secho(f"Starting cycle {manager.state.cycle_index}...", fg=typer.colors.BLUE)
        manager.run_cycle()
        typer.secho("Cycle step completed successfully.", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"Workflow failed: {e}", fg=typer.colors.RED)
        # Log full traceback
        logger.exception("Workflow execution failed")
        raise typer.Exit(code=1) from e
