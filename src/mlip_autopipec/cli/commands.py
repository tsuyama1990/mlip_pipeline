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
        # Cycle 02 Defaults
        "structure_gen": {
            "element": "Si",
            "crystal_structure": "diamond",
            "lattice_constant": 5.43,
            "supercell": [3, 3, 3]
        },
        "lammps": {
            "command": "lmp_serial",
            "timeout": 3600.0
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
    """Run the One-Shot Pipeline (Cycle 02)."""
    if not config_path.exists():
        typer.secho(f"Config file {config_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        config = Config.from_yaml(config_path)
        logging_infra.setup_logging(config.logging)
        logger = logging.getLogger("mlip_autopipec")

        # 1. Generate Structure
        typer.secho("Generating structure...", fg=typer.colors.BLUE)
        gen = StructureGenerator()
        structure = gen.generate(config.structure_gen)
        typer.secho("Structure generated.", fg=typer.colors.GREEN)

        # 2. Prepare Work Dir
        work_dir = Path("_work_md")
        work_dir.mkdir(exist_ok=True)
        job_dir = work_dir / "job_oneshot"
        job_dir.mkdir(exist_ok=True)

        # 3. Write Inputs
        typer.secho("Writing LAMMPS inputs...", fg=typer.colors.BLUE)
        io.write_lammps_data(structure, job_dir / "data.lammps")

        # Write basic in.lammps
        in_lammps_content = """
units metal
atom_style atomic
boundary p p p
read_data data.lammps
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
mass * 28.085
velocity all create 300.0 12345
fix 1 all nvt temp 300.0 300.0 0.1
dump 1 all custom 100 dump.lammpstrj id type x y z
run 100
"""
        (job_dir / "in.lammps").write_text(in_lammps_content)

        # 4. Run LAMMPS
        typer.secho(f"Running LAMMPS ({config.lammps.command})...", fg=typer.colors.BLUE)
        start_time = time.time()
        cmd = f"{config.lammps.command} -in in.lammps"

        status = JobStatus.PENDING
        try:
            io.run_subprocess(cmd, timeout=config.lammps.timeout, cwd=job_dir)
            status = JobStatus.COMPLETED
        except RuntimeError as e:
            logger.error(f"LAMMPS failed: {e}")
            status = JobStatus.FAILED

        duration = time.time() - start_time

        # 5. Parse Output
        final_struct = None
        if status == JobStatus.COMPLETED:
            try:
                # Assuming single element for now
                final_struct = io.read_lammps_dump(
                    job_dir / "dump.lammpstrj",
                    species=[config.structure_gen.element]
                )
            except Exception as e:
                logger.error(f"Parsing failed: {e}")
                status = JobStatus.FAILED

        result = LammpsResult(
            job_id="oneshot",
            status=status,
            work_dir=job_dir,
            duration_seconds=duration,
            final_structure=final_struct,
            trajectory_path=job_dir / "dump.lammpstrj"
        )
        logger.info(f"Cycle 02 Result: {result}")

        if status == JobStatus.COMPLETED:
            typer.secho("Cycle 02 Completed: Status DONE", fg=typer.colors.GREEN)
            if final_struct:
                typer.secho(f"Final Structure: {final_struct.get_chemical_formula()}")
        else:
            typer.secho("Cycle 02 Failed.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
