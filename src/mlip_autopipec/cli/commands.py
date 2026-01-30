import logging
from pathlib import Path

import typer

from mlip_autopipec.domain_models.config import (
    Config,
    LammpsConfig,
    LoggingConfig,
    MDConfig,
    PotentialConfig,
    StructureGenConfig,
)
from mlip_autopipec.domain_models.job import JobStatus, LammpsResult
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

    # Instantiate models with default values
    config = Config(
        project_name="MyMLIPProject",
        logging=LoggingConfig(),
        potential=PotentialConfig(),
        lammps=LammpsConfig(),
        structure_gen=StructureGenConfig(),
        md=MDConfig(),
    )

    try:
        # Convert to dict, excluding defaults? No, we want explicit template.
        # But dumping defaults is good for template.
        # Use mode='json' to handle Path and Enums
        data = config.model_dump(mode="json")
        io.dump_yaml(data, path)
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
            typer.secho(
                f"Simulation Completed: Status {result.status.value}",
                fg=typer.colors.GREEN,
            )

            # Check for specific result attributes (LammpsResult)
            if isinstance(result, LammpsResult):
                logging.getLogger("mlip_autopipec").info(
                    f"Trajectory saved to: {result.trajectory_path}"
                )

            logging.getLogger("mlip_autopipec").info(
                f"Duration: {result.duration_seconds:.2f}s"
            )
        else:
            typer.secho(
                f"Simulation Failed: Status {result.status.value}", fg=typer.colors.RED
            )
            typer.echo(f"Log Tail:\n{result.log_content}")
            raise typer.Exit(code=1)

    except Exception as e:
        typer.secho(f"Execution failed: {e}", fg=typer.colors.RED)
        logging.getLogger("mlip_autopipec").exception("Execution failed")
        raise typer.Exit(code=1) from e
