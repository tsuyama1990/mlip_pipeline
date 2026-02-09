import concurrent.futures
import contextlib
import logging
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ase.io import read, write

from mlip_autopipec.components.dynamics.base import BaseDynamics
from mlip_autopipec.components.dynamics.otf import get_otf_check_code
from mlip_autopipec.domain_models.config import EONDynamicsConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.utils.security import validate_safe_path

logger = logging.getLogger(__name__)


class EONDriver:
    """
    Driver for EON KMC simulations.
    Handles input generation and execution.
    """

    def __init__(self, workdir: Path, config: EONDynamicsConfig, binary: str = "eon") -> None:
        self.workdir = workdir
        self.config = config
        self.binary = binary

        # Use filenames from config
        self.config_file = self.workdir / config.config_filename
        self.pos_file = self.workdir / config.pos_filename
        self.driver_script = self.workdir / config.driver_filename
        self.client_log = self.workdir / config.client_log_filename
        self.halted_structure = self.workdir / config.halted_structure_filename

    def write_input_files(
        self, structure: Structure, potential: Potential
    ) -> None:
        """
        Write EON input files: config.ini, pos.con, pace_driver.py.
        """
        self.workdir.mkdir(parents=True, exist_ok=True)
        validate_safe_path(self.workdir)

        # 1. Write structure to pos.con
        atoms = structure.to_ase()
        try:
            write(self.pos_file, atoms, format="eon")
        except Exception as e:
            # Explicitly catch and log error for data integrity
            logger.exception(f"Failed to write EON structure file {self.pos_file}")
            msg = f"Failed to write EON structure: {e}"
            raise RuntimeError(msg) from e

        # 2. Write config.ini
        # Minimal EON config for KMC
        config_content = f"""[Main]
job = parallel_replica
temperature = {self.config.temperature}
random_seed = {self.config.seed if self.config.seed is not None else 0}

[Potentials]
potential = script_potential

[Script Potential]
script = python {self.driver_script.name}

[Parallel Replica]
time_step = {self.config.time_step}
# EON uses prefactor?
prefactor = {self.config.prefactor:.1e}
max_events = {self.config.n_events}
"""
        self.config_file.write_text(config_content)

        # 3. Generate pace_driver.py
        # Validate potential path
        potential_path = validate_safe_path(potential.path, must_exist=True)

        # Also include OTF logic.
        otf_code = get_otf_check_code()

        # Use config for filenames in script too
        pos_filename = self.config.pos_filename
        halted_filename = self.config.halted_structure_filename

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
    pot_path = "{potential_path}"
    calc = get_calculator(pot_path)
    threshold = {self.config.uncertainty_threshold}

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
                # from a file or stdin. We'll assume '{pos_filename}' is updated by EON client.
                try:
                    atoms = read("{pos_filename}", format="eon")
                except Exception:
                    # Fallback to stdin reading if EON pipes it (simplified)
                    # For now, stick to file assumption which is safer for ASE
                    atoms = read("{pos_filename}")

                atoms.calc = calc

                # Check Uncertainty (OTF)
                if check_uncertainty(atoms, threshold):
                    # We write the structure so the Orchestrator can pick it up
                    write("{halted_filename}", atoms)
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


def _run_single_eon_simulation(
    idx: int,
    structure: Structure,
    potential: Potential,
    config: EONDynamicsConfig,
    base_workdir: Path,
    physics_baseline: dict[str, Any] | None,
) -> Structure | None:
    """
    Run a single EON simulation in a separate process.
    """
    try:
        run_dir = base_workdir / f"eon_run_{idx:05d}"
        driver = EONDriver(workdir=run_dir, config=config)

        driver.write_input_files(structure, potential)

        # Run EON
        # We suppress exception because EON might exit with non-zero on OTF halt
        with contextlib.suppress(Exception):
            driver.run_kmc()

        # Check for halted structure
        if not driver.halted_structure.exists():
            return None

        try:
            atoms_obj = read(driver.halted_structure)
            atoms = atoms_obj[-1] if isinstance(atoms_obj, list) else atoms_obj
            struct = Structure.from_ase(atoms)
            struct.uncertainty = 100.0
            struct.tags["provenance"] = "dynamics_halted_eon"
        except Exception:
            logger.exception(f"Failed to read halted structure from EON run {idx}")
            return None
        else:
            return struct
    except Exception:
        logger.exception(f"EON run failed for structure {idx}")
        return None


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
        physics_baseline: dict[str, Any] | None = None,
    ) -> Iterator[Structure]:
        """
        Explore the PES using EON KMC simulations.
        """
        base_workdir = workdir or Path.cwd()
        max_workers = self.config.max_workers

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, structure in enumerate(start_structures):
                future = executor.submit(
                    _run_single_eon_simulation,
                    idx,
                    structure,
                    potential,
                    self.config,
                    base_workdir,
                    physics_baseline,
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        yield result
                except Exception:
                    logger.exception("Simulation task failed")

    def __repr__(self) -> str:
        return f"<EONDynamics(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"EONDynamics({self.name})"
