"""EON kMC wrapper."""

import shutil
import subprocess
from pathlib import Path

from ase import Atoms
from ase.io import write
from loguru import logger

from pyacemaker.core.config import EONConfig
from pyacemaker.core.exceptions import DynamicsError


class EONWrapper:
    """Wrapper for EON kMC software."""

    def __init__(self, config: EONConfig) -> None:
        """Initialize EON wrapper."""
        self.config = config
        self.logger = logger.bind(name="EONWrapper")

    def _validate_executable(self) -> Path:
        """Validate EON executable."""
        exe = shutil.which(self.config.executable)
        if not exe:
            msg = f"EON executable not found: {self.config.executable}"
            raise FileNotFoundError(msg)
        return Path(exe)

    def run_search(self, atoms: Atoms, potential_path: Path, work_dir: Path) -> None:
        """Run EON search."""
        work_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Setting up EON calculation in {work_dir}")

        # Validate inputs
        if len(atoms) == 0:
            raise DynamicsError("Input atoms object is empty.")
        if not atoms.pbc.any():
            self.logger.warning("EON usually requires periodic boundary conditions.")

        # Validate executable first
        exe_path = self._validate_executable()

        # 1. Write structure (pos.con is standard for EON)
        try:
            write(work_dir / "pos.con", atoms, format="eon")
        except Exception:
            # Fallback or assume extension handling works
            self.logger.warning("Could not write 'eon' format, trying generic write.")
            write(work_dir / "pos.con", atoms)

        # 2. Generate config.ini with parameters injection
        # Use defaults if not provided in config
        params = self.config.parameters
        main_job = params.get("job", "process_search")
        temperature = params.get("temperature", 300.0)
        seed = params.get("random_seed", 12345)
        converged_force = params.get("converged_force", 0.01)
        saddle_method = params.get("saddle_method", "min_mode")

        config_content = f"""[Main]
job = {main_job}
temperature = {temperature}
random_seed = {seed}

[Potential]
potential = script
script_path = ./pace_driver.py

[Optimizer]
converged_force = {converged_force}

[Saddle Search]
method = {saddle_method}
"""
        (work_dir / "config.ini").write_text(config_content)

        # 3. Generate pace_driver.py
        # Driver that uses ASE to calculate energy/forces with potential
        # We need to make sure this script can run in the environment where EON runs.
        # It assumes ASE and pyacemaker are installed or PYTHONPATH is set correctly.

        # Use absolute path for potential
        pot_path_abs = potential_path.resolve()

        driver_content = f"""#!/usr/bin/env python3
import sys
from pathlib import Path
from ase.io import read

def create_calculator(potential_path: Path, atoms):
    # Try attaching real PACE calculator
    try:
        from ase.calculators.lammpslib import LAMMPSlib

        pot_path_str = str(potential_path)
        elements = sorted(list(set(atoms.get_chemical_symbols())))
        elem_str = " ".join(elements)

        cmds = [
            "pair_style pace",
            f"pair_coeff * * {{pot_path_str}} {{elem_str}}",
        ]

        # Need to handle command line invocation or direct instantiation
        return LAMMPSlib(lammps_header=cmds, log_file="lammps_driver.log")

    except (ImportError, RuntimeError):
        # Fallback to EMT
        from ase.calculators.emt import EMT
        return EMT()

def main():
    # 1. Read structure from EON format (usually pos.con is updated by EON?)
    # Or EON passes coordinates via stdin?
    # Documentation for EON script potential says it executes script.
    # Usually script reads 'pos.con' in current directory.

    try:
        atoms = read("pos.con", format="eon")
    except:
        # Fallback
        atoms = read("pos.con")

    # 2. Attach Potential
    potential_path = Path("{pot_path_abs}")
    atoms.calc = create_calculator(potential_path, atoms)

    # 3. Compute
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # 4. Output in EON format
    # Energy
    print(f"{{energy:.6f}}")
    # Forces (flat list or one per line?)
    # EON expects forces?
    # Assuming standard EON script potential:
    # Line 1: Energy
    # Line 2..N: Forces x y z

    for f in forces:
        print(f"{{f[0]:.6f}} {{f[1]:.6f}} {{f[2]:.6f}}")

if __name__ == "__main__":
    main()
"""
        driver_path = work_dir / "pace_driver.py"
        driver_path.write_text(driver_content)
        driver_path.chmod(0o755)

        # 4. Run eonclient
        cmd = [str(exe_path)]
        self.logger.info(f"Executing: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, cwd=work_dir, check=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            self.logger.exception("EON execution failed")
            msg = "EON execution failed"
            raise DynamicsError(msg) from e
        except FileNotFoundError as e:
            self.logger.exception("EON executable not found")
            msg = f"EON executable not found: {self.config.executable}"
            raise DynamicsError(msg) from e
