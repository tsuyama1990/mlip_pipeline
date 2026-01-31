import logging
from pathlib import Path
from typing import Literal, Optional, cast

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
    BulkStructureGenConfig,
    Config,
    LoggingConfig,
    PotentialConfig,
)
from mlip_autopipec.domain_models.dynamics import LammpsResult, MDConfig
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.infrastructure import io
from mlip_autopipec.infrastructure import logging as logging_infra
from mlip_autopipec.orchestration.workflow import run_one_shot
from mlip_autopipec.physics.training.dataset import DatasetManager
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner
from mlip_autopipec.physics.validation.runner import ValidationRunner


def init_project(path: Path) -> None:
    """
    Logic for initializing a new project.
    """
    if path.exists():
        typer.secho(f"File {path} already exists.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Cast log level to Literal
    log_level = cast(Literal["DEBUG", "INFO", "WARNING", "ERROR"], DEFAULT_LOG_LEVEL)

    # Create default configuration using Pydantic models
    # This ensures consistency with the schema and leverages defaults
    config = Config(
        project_name=DEFAULT_PROJECT_NAME,
        logging=LoggingConfig(level=log_level, file_path=Path(DEFAULT_LOG_FILENAME)),
        potential=PotentialConfig(
            elements=DEFAULT_ELEMENTS,
            cutoff=DEFAULT_CUTOFF,
            seed=DEFAULT_SEED,
        ),
        structure_gen=BulkStructureGenConfig(
            strategy="bulk",
            element="Si",
            crystal_structure="diamond",
            lattice_constant=5.43,
            rattle_stdev=0.1,
            supercell=(1, 1, 1),
        ),
        md=MDConfig(
            temperature=300.0,
            n_steps=1000,
            timestep=0.001,
            ensemble="NVT",
        ),
    )

    try:
        # Dump to YAML (using mode='json' to handle Paths, then to YAML)
        # Note: io.dump_yaml handles dicts. We convert model to dict.
        # We use exclude_none=True to avoid cluttering with optional None fields
        data = config.model_dump(mode="json", exclude_none=True)
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

            # Check for specific result attributes
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


def validate_model(
    config_path: Path, potential_path: Path, structure_path: Optional[Path] = None
) -> None:
    """
    Run physical validation tests on a potential.
    """
    if not config_path.exists():
        typer.secho(f"Config file {config_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if not potential_path.exists():
        typer.secho(f"Potential file {potential_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        config = Config.from_yaml(config_path)
        logging_infra.setup_logging(config.logging)

        # Load structure
        structure: Structure
        if structure_path:
            if not structure_path.exists():
                typer.secho(
                    f"Structure file {structure_path} not found.", fg=typer.colors.RED
                )
                raise typer.Exit(code=1)
            # Load first structure
            # type: ignore[no-untyped-call]
            s_iter = io.load_structures(structure_path)
            try:
                structure = next(s_iter)
            except StopIteration:
                typer.secho(
                    f"Structure file {structure_path} is empty.", fg=typer.colors.RED
                )
                raise typer.Exit(code=1)
        else:
            # Generate from config
            typer.echo("No structure provided. Generating from structure_gen config.")
            from mlip_autopipec.physics.structure_gen.generator import (
                StructureGenFactory,
            )

            gen = StructureGenFactory.get_generator(config.structure_gen)
            structure = gen.generate(config.structure_gen)

        logger = logging.getLogger("mlip_autopipec")
        logger.info(
            f"Validating potential {potential_path} using structure {structure.get_chemical_formula()}"
        )

        runner = ValidationRunner(config.validation)
        result = runner.validate(potential_path, structure)

        if result.overall_status == "FAIL":
            typer.secho(
                f"Validation Failed. Status: {result.overall_status}",
                fg=typer.colors.RED,
            )
        elif result.overall_status == "WARN":
            typer.secho(
                f"Validation Warning. Status: {result.overall_status}",
                fg=typer.colors.YELLOW,
            )
        else:
            typer.secho(
                f"Validation Completed. Status: {result.overall_status}",
                fg=typer.colors.GREEN,
            )

        typer.echo(f"Report generated at: {config.validation.report_path}")

    except Exception as e:
        typer.secho(f"Execution failed: {e}", fg=typer.colors.RED)
        logging.getLogger("mlip_autopipec").exception("Execution failed")
        raise typer.Exit(code=1) from e


def train_model(config_path: Path, dataset_path: Path) -> None:
    """
    Logic for training a potential.
    """
    if not config_path.exists():
        typer.secho(f"Config file {config_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if not dataset_path.exists():
        typer.secho(f"Dataset file {dataset_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        config = Config.from_yaml(config_path)
        logging_infra.setup_logging(config.logging)

        if config.training is None:
            typer.secho("Config must have a 'training' section.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        logger = logging.getLogger("mlip_autopipec")
        logger.info(f"Starting Training: {config.project_name}")

        # 1. Load structures
        logger.info(f"Loading structures from {dataset_path}")
        # type: ignore[no-untyped-call]
        structures = io.load_structures(dataset_path)

        # 2. Convert Dataset
        # We use a subdirectory for training data
        work_dir = Path("training_work")
        dataset_manager = DatasetManager(work_dir=work_dir / "data")
        pacemaker_dataset = dataset_manager.convert(
            structures, work_dir / "data" / "train.pckl.gzip"
        )

        # 3. Train
        runner = PacemakerRunner(
            work_dir=work_dir / "run",
            train_config=config.training,
            potential_config=config.potential,
        )

        result = runner.train(pacemaker_dataset)

        if result.status == JobStatus.COMPLETED:
            typer.secho("Training Completed", fg=typer.colors.GREEN)
            typer.echo(f"Potential saved to: {result.potential_path}")
            typer.echo(f"Metrics: {result.validation_metrics}")
            logger.info(f"Potential saved to: {result.potential_path}")
            logger.info(f"Metrics: {result.validation_metrics}")
        else:
            typer.secho("Training Failed", fg=typer.colors.RED)
            typer.echo(f"Log Tail:\n{result.log_content}")
            raise typer.Exit(code=1)

    except Exception as e:
        typer.secho(f"Execution failed: {e}", fg=typer.colors.RED)
        logging.getLogger("mlip_autopipec").exception("Execution failed")
        raise typer.Exit(code=1) from e
