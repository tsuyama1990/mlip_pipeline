import logging
from pathlib import Path
from typing import Literal, cast

import typer

from mlip_autopipec.constants import (
    DEFAULT_ACE_FS_PARAMS,
    DEFAULT_ACE_NDENSITY,
    DEFAULT_ACE_NPOT,
    DEFAULT_CUTOFF,
    DEFAULT_ELEMENTS,
    DEFAULT_LOG_FILENAME,
    DEFAULT_LOG_LEVEL,
    DEFAULT_PROJECT_NAME,
    DEFAULT_SEED,
    DEFAULT_STRUCT_STRATEGY,
    DEFAULT_STRUCT_ELEMENT,
    DEFAULT_STRUCT_CRYSTAL,
    DEFAULT_STRUCT_LATTICE,
    DEFAULT_STRUCT_RATTLE,
    DEFAULT_STRUCT_SUPERCELL,
    DEFAULT_MD_TEMP,
    DEFAULT_MD_STEPS,
    DEFAULT_MD_TIMESTEP,
    DEFAULT_MD_ENSEMBLE,
)
from mlip_autopipec.domain_models.config import (
    ACEConfig,
    BulkStructureGenConfig,
    Config,
    LoggingConfig,
    OrchestratorConfig,
    PotentialConfig,
)
from mlip_autopipec.domain_models.dynamics import LammpsResult, MDConfig
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.infrastructure import io
from mlip_autopipec.infrastructure import logging as logging_infra
from mlip_autopipec.orchestration.workflow import run_one_shot
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.physics.training.dataset import DatasetManager
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner
from mlip_autopipec.physics.validation.runner import ValidationRunner
from mlip_autopipec.physics.structure_gen.generator import StructureGenFactory


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
        orchestrator=OrchestratorConfig(
            max_iterations=5,
            uncertainty_threshold=5.0,
            halt_threshold=5,
            validation_frequency=1
        ),
        potential=PotentialConfig(
            elements=DEFAULT_ELEMENTS,
            cutoff=DEFAULT_CUTOFF,
            seed=DEFAULT_SEED,
            pair_style="hybrid/overlay",
            ace_params=ACEConfig(
                npot=DEFAULT_ACE_NPOT,
                fs_parameters=DEFAULT_ACE_FS_PARAMS,
                ndensity=DEFAULT_ACE_NDENSITY,
            ),
        ),
        structure_gen=BulkStructureGenConfig(
            strategy=cast(Literal["bulk"], DEFAULT_STRUCT_STRATEGY),
            element=DEFAULT_STRUCT_ELEMENT,
            crystal_structure=DEFAULT_STRUCT_CRYSTAL,
            lattice_constant=DEFAULT_STRUCT_LATTICE,
            rattle_stdev=DEFAULT_STRUCT_RATTLE,
            supercell=DEFAULT_STRUCT_SUPERCELL,
        ),
        md=MDConfig(
            temperature=DEFAULT_MD_TEMP,
            n_steps=DEFAULT_MD_STEPS,
            timestep=DEFAULT_MD_TIMESTEP,
            ensemble=cast(Literal["NVT", "NPT"], DEFAULT_MD_ENSEMBLE),
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

        # 1. Load structures (streaming)
        logger.info(f"Loading structures from {dataset_path}")
        # Use load_structures which uses iread internally (O(1) memory)
        # Note: DatasetManager.convert now expects iterator or list.
        # iread returns a generator.
        structures = io.load_structures(dataset_path)

        # 2. Convert Dataset
        # Use work_dir from TrainingConfig (Constitution compliance)
        work_dir = config.training.work_dir

        # Ensure it's absolute or relative to CWD correctly (Pydantic handles Path)
        # We can resolve it just in case
        work_dir = work_dir.resolve()

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

def validate_potential(config_path: Path, potential_path: Path) -> None:
    """
    Logic for validating a potential.
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

        # Use StructureGen to produce a perfect bulk structure for validation.
        gen_config = config.structure_gen

        # Use model_copy(update=...) to avoid mutating original config
        # Assuming BulkStructureGenConfig which has rattle_stdev
        if isinstance(gen_config, BulkStructureGenConfig):
            gen_config = gen_config.model_copy(update={"rattle_stdev": config.validation.validation_rattle_stdev})
        elif not isinstance(gen_config, BulkStructureGenConfig):
             # For validation, we prefer BulkStructureGenConfig if possible
             # But if user configured something else, we try to use it.
             # Warn if not bulk?
             pass

        generator = StructureGenFactory.get_generator(gen_config)
        structure = generator.generate(gen_config)

        runner = ValidationRunner(
            val_config=config.validation,
            pot_config=config.potential,
            potential_path=potential_path
        )

        typer.secho(f"Starting validation for {potential_path}", fg=typer.colors.BLUE)
        result = runner.validate(structure)

        if result.overall_status == "PASS":
            color = typer.colors.GREEN
        elif result.overall_status == "WARN":
            color = typer.colors.YELLOW
        else:
            color = typer.colors.RED

        typer.secho(f"Validation Finished: {result.overall_status}", fg=color)
        typer.echo("Metrics:")
        for m in result.metrics:
            status_icon = "✓" if m.passed else "✗"
            typer.echo(f"  {status_icon} {m.name}: {m.value:.4f} ({m.message})")

        typer.echo(f"Report generated at: {config.validation.report_path}")

    except Exception as e:
        typer.secho(f"Validation execution failed: {e}", fg=typer.colors.RED)
        logging.getLogger("mlip_autopipec").exception("Validation failed")
        raise typer.Exit(code=1) from e

def run_loop_cmd(config_path: Path) -> None:
    """
    Logic for running the Autonomous Active Learning Loop (Cycle 06).
    """
    if not config_path.exists():
        typer.secho(f"Config file {config_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        config = Config.from_yaml(config_path)
        logging_infra.setup_logging(config.logging)

        # Initialize Orchestrator
        orchestrator = Orchestrator(config, work_dir=Path.cwd())

        # Run Loop
        orchestrator.run_loop()

        typer.secho("Autonomous Loop Finished.", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"Execution failed: {e}", fg=typer.colors.RED)
        logging.getLogger("mlip_autopipec").exception("Execution failed")
        raise typer.Exit(code=1) from e
