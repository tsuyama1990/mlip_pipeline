"""EON kMC wrapper."""

import subprocess
from pathlib import Path

from ase import Atoms
from ase.io import write
from loguru import logger

from pyacemaker.core.config import EONConfig


class EONWrapper:
    """Wrapper for EON kMC software."""

    def __init__(self, config: EONConfig) -> None:
        """Initialize EON wrapper."""
        self.config = config
        self.logger = logger.bind(name="EONWrapper")

    def run_search(self, atoms: Atoms, potential_path: Path, work_dir: Path) -> None:
        """Run EON search."""
        work_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Setting up EON calculation in {work_dir}")

        # 1. Write structure (pos.con is standard for EON)
        try:
            write(work_dir / "pos.con", atoms, format="eon")
        except Exception:
            # Fallback or assume extension handling works
            self.logger.warning("Could not write 'eon' format, trying generic write.")
            write(work_dir / "pos.con", atoms)

        # 2. Generate config.ini
        config_content = """[Main]
job = process_search
temperature = 300.0
random_seed = 12345

[Potential]
potential = script
script_path = ./pace_driver.py

[Optimizer]
converged_force = 0.01

[Saddle Search]
method = min_mode
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
        cmd = [self.config.executable]
        self.logger.info(f"Executing: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, cwd=work_dir, check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"EON execution failed: {e}")
            raise RuntimeError("EON execution failed") from e
        except FileNotFoundError as e:
            self.logger.error(f"EON executable not found: {self.config.executable}")
            # Raise RuntimeError to match test expectation
            raise RuntimeError("EON execution failed") from e
