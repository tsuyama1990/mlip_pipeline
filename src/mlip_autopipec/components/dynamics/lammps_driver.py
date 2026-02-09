import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ase.io import read, write

from mlip_autopipec.components.dynamics.hybrid import generate_pair_style
from mlip_autopipec.components.dynamics.otf import generate_lammps_otf_commands
from mlip_autopipec.domain_models.config import LAMMPSDynamicsConfig, PhysicsBaselineConfig
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
        self,
        structure: Structure,
        potential: Potential,
        config: LAMMPSDynamicsConfig,
        physics_baseline: dict[str, Any] | None = None,
    ) -> None:
        """
        Write LAMMPS input and data files.

        Args:
            structure: Structure to simulate.
            potential: Potential to use.
            config: Dynamics configuration.
            physics_baseline: Optional physics baseline configuration.
        """
        self.workdir.mkdir(parents=True, exist_ok=True)

        # 1. Write structure to data.lammps
        atoms = structure.to_ase()
        write(self.data_file, atoms, format="lammps-data")

        # 2. Generate pair_style and pair_coeff (Hybrid/OTF aware)
        baseline_config = None
        if physics_baseline:
            baseline_config = PhysicsBaselineConfig.model_validate(physics_baseline)

        pair_style, pair_coeff = generate_pair_style(potential, baseline_config)

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
            pass

        try:
            # We redirect output to log file, but LAMMPS also writes to log.lammps by default.
            # Using -log explicitly.
            cmd = [self.binary, "-in", str(self.input_file), "-log", str(self.log_file)]
            result = subprocess.run(  # noqa: S603
                cmd, cwd=self.workdir, capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                logger.error(f"LAMMPS failed: {result.stderr}")
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
                types = atoms.get_array("type")
                # Map 1-based index to species
                species_map = {i + 1: s for i, s in enumerate(potential.species)}

                # Convert symbols
                from ase.data import atomic_numbers

                new_numbers = [atomic_numbers[species_map[t]] for t in types]
                atoms.set_atomic_numbers(new_numbers)
            except Exception as e:
                logger.warning(f"Could not map species from potential: {e}")

        return Structure.from_ase(atoms)
