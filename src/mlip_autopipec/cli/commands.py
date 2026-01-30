import logging
import shutil
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
from mlip_autopipec.domain_models.structure import Structure
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
    Run Cycle 02: One-Shot Pipeline.
    """
    if not config_path.exists():
        typer.secho(f"Config file {config_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        config = Config.from_yaml(config_path)
        logging_infra.setup_logging(config.logging)

        # 1. Check Executable
        if not shutil.which(config.lammps.command):
            typer.secho(
                f"Executable '{config.lammps.command}' not found. Please check your config.",
                fg=typer.colors.RED
            )
            raise typer.Exit(code=1)

        # 2. Generate Structure
        typer.secho("Generating structure...", fg=typer.colors.BLUE)
        gen = StructureGenerator(config.structure_gen)
        structure = gen.build_initial_structure()

        # 3. Setup Work Dir
        work_dir = Path("_work_md")
        work_dir.mkdir(exist_ok=True)
        job_dir = work_dir / "job_oneshot"
        if job_dir.exists():
            shutil.rmtree(job_dir)
        job_dir.mkdir()

        # 4. Write Inputs
        data_path = job_dir / "data.lammps"
        io.write_lammps_data(structure, data_path)

        input_path = job_dir / "in.lammps"
        _write_lammps_input(input_path, config.potential.cutoff)

        # 5. Run LAMMPS
        cmd = [config.lammps.command, "-in", "in.lammps"]
        typer.secho(f"Running LAMMPS in {job_dir}...", fg=typer.colors.BLUE)

        stdout, stderr = io.run_subprocess(
            cmd, cwd=job_dir, timeout=config.lammps.timeout
        )

        # 6. Parse Output
        dump_path = job_dir / "dump.lammpstrj"
        if dump_path.exists():
            traj = io.read_lammps_dump(dump_path)
            if traj:
                final_struct = Structure.from_ase(traj[-1])
                typer.secho("Simulation Completed: Status DONE", fg=typer.colors.GREEN)
                typer.echo(f"Final Energy: {final_struct.properties.get('energy', 'N/A')}")
            else:
                 typer.secho("Simulation Completed but trajectory is empty.", fg=typer.colors.YELLOW)
        else:
            typer.secho("Simulation Completed but no dump file found.", fg=typer.colors.YELLOW)

    except Exception as e:
        typer.secho(f"Simulation failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


def _write_lammps_input(path: Path, cutoff: float) -> None:
    """Helper to write a simple LJ LAMMPS input file."""
    content = f"""
units metal
atom_style atomic
boundary p p p

read_data data.lammps

pair_style lj/cut {cutoff}
pair_coeff * * 1.0 1.0

thermo 10
dump 1 all custom 1 dump.lammpstrj id type x y z fx fy fz
run 10
"""
    path.write_text(content)
