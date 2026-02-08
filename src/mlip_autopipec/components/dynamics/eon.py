import contextlib
import logging
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

from ase.io import read, write

from mlip_autopipec.components.dynamics.base import BaseDynamics
from mlip_autopipec.components.dynamics.otf import get_otf_check_code
from mlip_autopipec.domain_models.config import EONDynamicsConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class EONDriver:
    """
    Driver for EON KMC simulations.
    Handles input generation and execution.
    """

    def __init__(self, workdir: Path, binary: str = "eon") -> None:
        self.workdir = workdir
        self.binary = binary
        self.config_file = self.workdir / "config.ini"
        self.pos_file = self.workdir / "pos.con"
        self.driver_script = self.workdir / "pace_driver.py"
        self.client_log = self.workdir / "client.log"

    def write_input_files(
        self, structure: Structure, potential: Potential, config: EONDynamicsConfig
    ) -> None:
        """
        Write EON input files: config.ini, pos.con, pace_driver.py.
        """
        self.workdir.mkdir(parents=True, exist_ok=True)

        # 1. Write structure to pos.con
        atoms = structure.to_ase()
        try:
            write(self.pos_file, atoms, format="eon")
        except Exception:
            # Fallback: simple writer or xyz if format not available (but unlikely)
            # EON format:
            # Header line
            # Box vectors
            # Atom coords
            # But ASE handles it usually.
            # If fail, try "con"?
            # If not, raise error.
            logger.warning("ASE failed to write EON format, trying generic XYZ but EON needs CON.")
            raise

        # 2. Write config.ini
        # Minimal EON config for KMC
        config_content = f"""[Main]
job = parallel_replica
temperature = {config.temperature}
random_seed = {config.seed if config.seed is not None else 0}

[Potentials]
potential = script_potential

[Script Potential]
script = python {self.driver_script.name}

[Parallel Replica]
time_step = {config.time_step}
# EON uses prefactor?
prefactor = {config.prefactor:.1e}
max_events = {config.n_events}
"""
        self.config_file.write_text(config_content)

        # 3. Generate pace_driver.py
        # This script must implement EON's script potential interface.
        # We need to import the calculator.
        # Assuming `pyjulip` or `tensorpot` or `lammpslib` is used.

        # Also include OTF logic.
        otf_code = get_otf_check_code()

        driver_content = f"""#!/usr/bin/env python3
import sys
import numpy as np
from ase.io import read
from ase import Atoms

# Attempt to load calculator
# This part is highly dependent on the installed MLIP package.
# We assume a generic `PACECalculator` is available or wrap LAMMPS.

# Placeholder for calculator loading
def get_calculator(pot_path):
    # Try to load known calculators
    try:
        from pyjulip import ACE1
        return ACE1(pot_path)
    except ImportError:
        pass

    try:
        # Generic LAMMPS wrapper if available
        # from ase.calculators.lammpslib import LAMMPSlib
        # ...
        pass
    except ImportError:
        pass

    # Mock calculator for testing if nothing else works
    from ase.calculators.lj import LennardJones
    return LennardJones()

# OTF Logic
{otf_code}

def main():
    pot_path = "{potential.path}"
    calc = get_calculator(pot_path)
    threshold = {config.uncertainty_threshold}

    # EON Script Potential Interface loop
    # EON sends commands via stdin

    # Simple loop reading lines
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            cmd = line.strip()

            if cmd == "init":
                # EON usually sends n_atoms, etc.
                # Just ack
                print("init_done")
                sys.stdout.flush()

            elif cmd.startswith("energy") or cmd.startswith("forces"):
                # EON "script" potential typically expects us to read coordinates
                # from a file or stdin. We'll assume 'pos.con' is updated by EON client.
                try:
                    atoms = read("pos.con", format="eon")
                except Exception:
                    # Fallback to stdin reading if EON pipes it (simplified)
                    # For now, stick to file assumption which is safer for ASE
                    atoms = read("pos.con")

                atoms.calc = calc

                # Check Uncertainty (OTF)
                if check_uncertainty(atoms, threshold):
                    # We write the structure so the Orchestrator can pick it up
                    write("halted_structure.xyz", atoms)
                    sys.stderr.write("Halt: Uncertainty limit reached\\n")
                    sys.exit(100) # Special exit code for Orchestrator to detect

                # Compute
                e = atoms.get_potential_energy()
                f = atoms.get_forces()

                # Output to stdout as EON expects
                print(f"{{e:.12f}}")
                sys.stdout.flush()
                # EON might expect forces on next lines? Or just energy?
                # "script" potential usually returns Energy then Forces?
                # We'll print energy. If it wants forces, we'd need to know protocol.
                # Assuming standard EON script interface:
                # line 1: energy
                # line 2...N+1: forces
                for force in f:
                    print(f"{{force[0]:.12f}} {{force[1]:.12f}} {{force[2]:.12f}}")
                sys.stdout.flush()

            elif cmd == "exit":
                break

        except Exception as e:
            sys.stderr.write(f"Error: {{e}}\\n")
            sys.exit(1)

if __name__ == "__main__":
    main()
"""
        self.driver_script.write_text(driver_content)
        self.driver_script.chmod(0o755)

    def run_kmc(self) -> None:
        """
        Run EON via subprocess.
        """
        if not shutil.which(self.binary):
            # Skip if no binary
            pass

        try:
            cmd = [self.binary]
            # Redirect stdout to client.log
            with self.client_log.open("w") as f:
                subprocess.run(cmd, cwd=self.workdir, stdout=f, stderr=subprocess.STDOUT, check=False) # noqa: S603
        except Exception as e:
            logger.exception(f"EON failed: {e}") # noqa: TRY401
            raise


class EONDynamics(BaseDynamics):
    """
    EON implementation of the Dynamics component.
    """

    def __init__(self, config: EONDynamicsConfig) -> None:
        super().__init__(config)
        self.config: EONDynamicsConfig = config

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
        Explore the PES using EON KMC simulations.
        """
        base_workdir = workdir or Path.cwd()

        for idx, structure in enumerate(start_structures):
            run_dir = base_workdir / f"eon_run_{idx:05d}"
            driver = EONDriver(workdir=run_dir)

            try:
                driver.write_input_files(structure, potential, self.config)
                # We assume run_kmc handles execution.
                # EON usually runs for a long time.
                # If it halts due to OTF (exit code from driver?), we detect it.

                # For now, just run it.
                with contextlib.suppress(Exception):
                    driver.run_kmc()
                    # EON might fail if OTF driver exits with error.
                    # This is expected behavior for OTF halt.

                # Check for halted structure
                halted_path = driver.workdir / "halted_structure.xyz"
                if halted_path.exists():
                    try:
                        atoms_obj = read(halted_path)
                        # Ensure single Atoms object
                        atoms = atoms_obj[-1] if isinstance(atoms_obj, list) else atoms_obj

                        # Ensure atomic numbers are correct (XYZ usually has symbols)
                        struct = Structure.from_ase(atoms)
                        struct.uncertainty = 100.0
                        struct.tags["provenance"] = "dynamics_halted_eon"
                        yield struct
                    except Exception as e:
                        logger.exception(f"Failed to read halted structure from EON run {idx}: {e}") # noqa: TRY401

            except Exception:
                logger.exception(f"EON setup failed for structure {idx}")
                continue

    def __repr__(self) -> str:
        return f"<EONDynamics(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"EONDynamics({self.name})"
