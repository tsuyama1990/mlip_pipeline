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
from mlip_autopipec.domain_models.config import (
    Config,
    DFTConfig,
    LammpsConfig,
    LoggingConfig,
    MDConfig,
    PotentialConfig,
    StructureGenConfig,
)
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.infrastructure import io
from mlip_autopipec.infrastructure import logging as logging_infra
from mlip_autopipec.orchestration.workflow import run_one_shot


def init_project(path: Path) -> None:
    """
    Logic for initializing a new project.
    """
    if path.exists():
        typer.secho(f"File {path} already exists.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Use defaults from domain models
    # Note: We construct a dictionary that mimics the structure,
    # but populated with defaults from the models where possible.
    # However, some fields like 'pseudopotentials' are mandatory and don't have defaults.

    # We create a sample valid config.

    # Explicitly cast DEFAULT_LOG_LEVEL to Literal if needed, but Pydantic should handle string matching.
    # Mypy complains strict type mismatch.
    from typing import Literal, cast

    log_level = cast(Literal["DEBUG", "INFO", "WARNING", "ERROR"], DEFAULT_LOG_LEVEL)

    template = Config(
        project_name=DEFAULT_PROJECT_NAME,
        logging=LoggingConfig(
            level=log_level,
            file_path=Path(DEFAULT_LOG_FILENAME)
        ),
        potential=PotentialConfig(
            elements=DEFAULT_ELEMENTS,
            cutoff=DEFAULT_CUTOFF,
            seed=DEFAULT_SEED
        ),
        structure_gen=StructureGenConfig(
            strategy="bulk",
            element="Si",
            crystal_structure="diamond",
            lattice_constant=5.43,
            rattle_stdev=0.1,
            supercell=(1, 1, 1)
        ),
        md=MDConfig(
            temperature=300.0,
            n_steps=1000,
            timestep=0.001,
            ensemble="NVT"
        ),
        lammps=LammpsConfig(), # Uses defaults
        dft=DFTConfig(
            pseudopotentials={"Si": Path("Si.pbe-n-kjpaw_psl.1.0.0.UPF")}
        )
    )

    try:
        # Dump to YAML via dict
        io.dump_yaml(template.model_dump(mode="json"), path)
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


def run_cycle_02_cmd(config_path: Path) -> None:
    """
    Logic for running the Cycle 02 One-Shot Pipeline.
    """
    if not config_path.exists():
        typer.secho(f"Config file {config_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        config = Config.from_yaml(config_path)
        logging_infra.setup_logging(config.logging)

        result = run_one_shot(config)

        if result.status == JobStatus.COMPLETED:
            typer.secho(f"Simulation Completed: Status {result.status.value}", fg=typer.colors.GREEN)

            # Check for specific result attributes (LammpsResult)
            if hasattr(result, "trajectory_path"):
                logging.getLogger("mlip_autopipec").info(f"Trajectory saved to: {result.trajectory_path}") # type: ignore[attr-defined]

            logging.getLogger("mlip_autopipec").info(f"Duration: {result.duration_seconds:.2f}s")
        else:
            typer.secho(f"Simulation Failed: Status {result.status.value}", fg=typer.colors.RED)
            typer.echo(f"Log Tail:\n{result.log_content}")
            raise typer.Exit(code=1)

    except Exception as e:
        typer.secho(f"Execution failed: {e}", fg=typer.colors.RED)
        logging.getLogger("mlip_autopipec").exception("Execution failed")
        raise typer.Exit(code=1) from e
