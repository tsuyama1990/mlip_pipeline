import logging
import time
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
from mlip_autopipec.domain_models.structure import JobStatus, LammpsResult
from mlip_autopipec.infrastructure import io
from mlip_autopipec.infrastructure import logging as logging_infra
from mlip_autopipec.modules.structure_gen.generator import StructureGenerator


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
        # Add Cycle 02 defaults to template
        "lammps": {
            "command": "lmp_serial",
            "cores": 1,
            "timeout": 3600.0
        },
        "structure_gen": {
            "element": "Si",
            "crystal_structure": "diamond",
            "lattice_constant": 5.43,
            "supercell": [1, 1, 1],
            "rattle_stdev": 0.01
        }
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


def run_cycle_02(config_path: Path) -> None:
    """
    Run the One-Shot Pipeline (Cycle 02).
    """
    if not config_path.exists():
        typer.secho(f"Config file {config_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        config = Config.from_yaml(config_path)
        logging_infra.setup_logging(config.logging)
    except Exception as e:
        typer.secho(f"Failed to load config: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    # 1. Generate Structure
    typer.secho("Generating structure...", fg=typer.colors.BLUE)
    try:
        gen = StructureGenerator(config.structure_gen, seed=config.potential.seed)
        structure = gen.build()
        typer.secho(f"Generated structure with {len(structure.symbols)} atoms.", fg=typer.colors.BLUE)
    except Exception as e:
        typer.secho(f"Structure generation failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    # 2. Prepare Workspace
    work_dir = Path("_work_md/job_001")
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        typer.secho(f"Failed to create work directory: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    # 3. Write Inputs
    try:
        data_path = work_dir / "data.lammps"
        io.write_lammps_data(structure, data_path)

        input_path = work_dir / "in.lammps"
        dump_path = work_dir / "dump.lammpstrj"

        # Simple template for Cycle 02 testing
        # Using LJ potential (assuming Si matches LJ roughly or just for testing mechanism)
        # Assuming units metal.
        input_content = f"""
units metal
atom_style atomic
dimension 3
boundary p p p
read_data {data_path.name}

# Basic LJ for testing
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0

thermo 10
dump 1 all custom 1 {dump_path.name} id type x y z fx fy fz

fix 1 all nve
run 10
"""
        input_path.write_text(input_content)
    except Exception as e:
        typer.secho(f"Failed to write inputs: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    # 4. Run LAMMPS
    typer.secho("Running LAMMPS...", fg=typer.colors.BLUE)
    cmd = config.lammps.command.split() + ["-in", "in.lammps"]
    start_time = time.time()

    stdout = ""
    status = JobStatus.PENDING

    try:
        stdout, stderr = io.run_subprocess(cmd, timeout=config.lammps.timeout, cwd=work_dir)
        status = JobStatus.COMPLETED
    except TimeoutError:
        status = JobStatus.TIMEOUT
        typer.secho("Simulation timed out.", fg=typer.colors.RED)
    except Exception as e:
        status = JobStatus.FAILED
        typer.secho(f"Simulation failed: {e}", fg=typer.colors.RED)
        # Don't exit yet, create result

    duration = time.time() - start_time

    # 5. Parse Output
    final_structure = structure
    if status == JobStatus.COMPLETED:
        try:
            final_structure = io.read_lammps_dump(dump_path)
        except Exception as e:
            typer.secho(f"Failed to parse dump: {e}", fg=typer.colors.RED)
            status = JobStatus.FAILED

    result = LammpsResult(
        job_id="job_001",
        status=status,
        work_dir=work_dir,
        duration_seconds=duration,
        log_content=stdout[-1000:] if stdout else "",
        final_structure=final_structure,
        trajectory_path=dump_path
    )

    color = typer.colors.GREEN if status == JobStatus.COMPLETED else typer.colors.RED
    typer.secho(f"Simulation Completed. Status: {result.status}", fg=color)

    if status != JobStatus.COMPLETED:
        raise typer.Exit(code=1)
