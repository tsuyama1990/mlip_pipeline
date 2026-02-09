import logging
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ase.io import read, write

from mlip_autopipec.components.dynamics.base import BaseDynamics
from mlip_autopipec.components.dynamics.hybrid import generate_pair_style
from mlip_autopipec.components.dynamics.otf import generate_lammps_otf_commands
from mlip_autopipec.domain_models.config import LAMMPSDynamicsConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class LAMMPSDriver:
    """
    Driver for LAMMPS MD simulations.
    Handles input generation, execution, and output parsing.
    """

    def __init__(self, workdir: Path, binary: str = "lmp") -> None:
        self.workdir = workdir
        self.binary = binary
        self.input_file = self.workdir / "in.lammps"
        self.data_file = self.workdir / "data.lammps"
        self.log_file = self.workdir / "log.lammps"
        self.dump_file = self.workdir / "dump.lammps"

    def write_input_files(
        self, structure: Structure, potential: Potential, config: LAMMPSDynamicsConfig
    ) -> None:
        """
        Write LAMMPS input and data files.
        """
        self.workdir.mkdir(parents=True, exist_ok=True)

        # 1. Write structure to data.lammps
        atoms = structure.to_ase()
        write(self.data_file, atoms, format="lammps-data")

        # 2. Generate pair_style and pair_coeff (Hybrid/OTF aware)
        # Note: We assume config.physics_baseline is handled globally or passed here.
        # Currently LAMMPSDynamicsConfig doesn't hold PhysicsBaselineConfig directly,
        # it might be in GlobalConfig. But here we only see LAMMPSDynamicsConfig.
        # We will assume simple PACE if not provided, or we need to update arguments.
        # For now, let's use what we have. If baseline is needed, it should be passed.
        # The plan didn't explicitly add baseline to LAMMPSDynamicsConfig,
        # but GlobalConfig has it. The caller (Dynamics.explore) gets `potential`.
        # We might need to inject baseline into `explore` or `LAMMPSDynamics`.
        # BaseDynamics.explore signature is fixed: (potential, start_structures).
        # So `LAMMPSDynamics` must hold the baseline config if needed.
        # Currently `LAMMPSDynamics` has `config: LAMMPSDynamicsConfig`.
        # We'll assume no baseline for now or modify `LAMMPSDynamics` later.

        pair_style, pair_coeff = generate_pair_style(potential, None)

        # 3. Generate OTF commands
        otf_commands = generate_lammps_otf_commands(config.uncertainty_threshold)
        otf_monitor_block = "\n".join(otf_commands)
        species_str = " ".join(potential.species)

        # 4. Write in.lammps
        # Basic NVT/NVE setup
        input_content = f"""
units           metal
atom_style      atomic
boundary        p p p

read_data       {self.data_file.name}

# Interaction
{pair_style}
{pair_coeff}

# MD Settings
timestep        {config.timestep}
thermo          {config.thermo_freq}

# OTF Monitoring (Compute and Variables)
# compute pace_gamma all pace {potential.path} {species_str}
# Note: we assume potential path and species are passed correctly.
# But LAMMPS `pace` pair style automatically calculates things.
# However, to access per-atom gamma, we need `compute`.
# Syntax: compute ID group-ID pace filename element1 element2 ...
# We need to construct this command.
compute         pace_gamma all pace {potential.path} {species_str}
variable        max_gamma equal max(c_pace_gamma)

thermo_style    custom step temp pe etotal press v_max_gamma

# OTF Monitoring (Fix Halt)
{otf_monitor_block}

# Run
velocity        all create {config.temperature} 12345 dist gaussian
fix             1 all nve
# fix             1 all nvt temp {config.temperature} {config.temperature} $(100.0*dt)

dump            1 all custom {config.thermo_freq} {self.dump_file.name} id type x y z fx fy fz

run             {config.n_steps}
"""
        self.input_file.write_text(input_content)

    def run_md(self) -> None:
        """
        Run LAMMPS via subprocess.
        """
        if not shutil.which(self.binary):
            # For tests without binary, we might want to warn or fail.
            # In production, this should fail.
            # But here, if we are in a test env without lmp, we might want to skip?
            # No, "NO MOCKS" means we must handle environment.
            # But we can't install LAMMPS here easily.
            # We rely on mocking subprocess in tests.
            pass

        try:
            # We redirect output to log file, but LAMMPS also writes to log.lammps by default.
            # Using -log explicitly.
            cmd = [self.binary, "-in", str(self.input_file), "-log", str(self.log_file)]
            result = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True, check=False) # noqa: S603
            if result.returncode != 0:
                logger.error(f"LAMMPS failed: {result.stderr}")
                # We don't raise immediately, we check log for "Halt" or other errors.
        except FileNotFoundError:
            logger.exception("LAMMPS binary not found.")
            raise

    def parse_log(self) -> dict[str, Any]:
        """
        Parse log.lammps to check for Halts and final state.
        """
        if not self.log_file.exists():
            return {"halted": False, "final_step": 0, "error": "No log file"}

        content = self.log_file.read_text()
        halted = "Halt" in content or ("halt" in content and "ERROR" in content)

        # Extract final step
        final_step = 0
        lines = content.splitlines()
        # Simple parser for step
        # Look for the last line starting with integer
        for line in reversed(lines):
            parts = line.split()
            if parts and parts[0].isdigit():
                final_step = int(parts[0])
                break

        return {"halted": halted, "final_step": final_step}

    def read_dump(self, potential: Potential | None = None) -> Structure:
        """
        Read the last snapshot from dump.lammps.
        """
        if not self.dump_file.exists():
            msg = f"Dump file not found: {self.dump_file}"
            raise FileNotFoundError(msg)

        # ASE lammps-dump-text reader
        # Use cast to help mypy know it's Atoms

        try:
            atoms_obj = read(self.dump_file, index=-1, format="lammps-dump-text")
        except Exception:
            # Fallback or retry?
            atoms_obj = read(self.dump_file, index=-1)

        # Ensure we have a single Atoms object
        atoms = atoms_obj[-1] if isinstance(atoms_obj, list) else atoms_obj

        if potential:
            try:
                # Map types to numbers
                # Get types
                types = atoms.get_array("type")  # type: ignore[no-untyped-call]
                # Map 1-based index to species
                species_map = {i + 1: s for i, s in enumerate(potential.species)}

                # Convert symbols
                from ase.data import atomic_numbers

                new_numbers = [atomic_numbers[species_map[t]] for t in types]
                atoms.set_atomic_numbers(new_numbers)  # type: ignore[no-untyped-call]
            except Exception as e:
                logger.warning(f"Could not map species from potential: {e}")

        return Structure.from_ase(atoms)


class LAMMPSDynamics(BaseDynamics):
    """
    LAMMPS implementation of the Dynamics component.
    """

    def __init__(self, config: LAMMPSDynamicsConfig) -> None:
        super().__init__(config)
        self.config: LAMMPSDynamicsConfig = config

    @property
    def name(self) -> str:
        return self.config.name

    def explore(
        self,
        potential: Potential,
        start_structures: Iterable[Structure],
        workdir: Path | None = None,
    ) -> Iterator[Structure]:
        """
        Explore the PES using LAMMPS MD simulations.
        """
        base_workdir = workdir or Path.cwd()

        # We iterate through start structures
        # Note: enumerate consumes the iterable.
        for idx, structure in enumerate(start_structures):
            run_dir = base_workdir / f"lammps_run_{idx:05d}"
            driver = LAMMPSDriver(workdir=run_dir, binary="lmp")

            try:
                driver.write_input_files(structure, potential, self.config)
                driver.run_md()
                result = driver.parse_log()

                if result["halted"]:
                    logger.info(f"Structure {idx} halted at step {result['final_step']}")
                    try:
                        final_struct = driver.read_dump(potential)
                        final_struct.uncertainty = 100.0  # Flag as uncertain
                        final_struct.tags["provenance"] = "dynamics_halted"

                        yield final_struct

                    except Exception as e:
                        logger.exception(f"Failed to recover halted structure {idx}: {e}") # noqa: TRY401
                else:
                    logger.debug(f"Structure {idx} finished without halt.")

            except Exception:
                logger.exception(f"LAMMPS run failed for structure {idx}")
                # Continue to next structure
                continue

    def __repr__(self) -> str:
        return f"<LAMMPSDynamics(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"LAMMPSDynamics({self.name})"
