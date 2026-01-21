import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
from ase import Atoms

from mlip_autopipec.core.config import DFTConfig
from mlip_autopipec.core.exceptions import DFTError, DFTRuntimeError
from mlip_autopipec.core.models import DFTResult
from mlip_autopipec.dft.input_gen import write_pw_input
from mlip_autopipec.dft.parsers import parse_pw_output
from mlip_autopipec.dft.utils import get_kpoints, get_sssp_pseudopotentials, is_magnetic

logger = logging.getLogger(__name__)


class QERunner:
    """
    Executes Quantum Espresso (pw.x) calculations.
    """

    def __init__(self, config: DFTConfig) -> None:
        """
        Initializes the QE runner.

        Args:
            config: The DFT configuration.
        """
        self.config = config

    def run_static_calculation(self, atoms: Atoms, run_dir: Path) -> DFTResult:
        """
        Runs a static SCF calculation for the given structure.

        Args:
            atoms: The atomic structure to calculate.
            run_dir: The directory to run the calculation in.

        Returns:
            The calculation results (energy, forces, stress).

        Raises:
            DFTRuntimeError: If execution or parsing fails.
        """
        run_dir.mkdir(parents=True, exist_ok=True)

        input_file = run_dir / "pw.in"
        output_file = run_dir / "pw.out"

        self._generate_input(atoms, run_dir, input_file)
        self._execute_calculation(input_file, output_file, run_dir)
        result = self._parse_results(output_file)
        self._cleanup(run_dir)

        return result

    def _generate_input(self, atoms: Atoms, run_dir: Path, input_file: Path) -> None:
        """Generates the Quantum Espresso input file."""
        elements = sorted(set(atoms.get_chemical_symbols()))
        try:
            pseudos = get_sssp_pseudopotentials(elements, self.config.pseudopotential_dir)
        except Exception as e:
            msg = f"Failed to setup pseudopotentials: {e}"
            raise DFTRuntimeError(msg) from e

        kpts = get_kpoints(atoms, self.config.kpoints_density)

        # Handle Magnetism
        if is_magnetic(atoms):
            # Initialize random magnetic moments to break symmetry
            # We set them on the atoms object so ASE writes starting_magnetization
            magmoms = [1.0 + 0.1 * np.random.random() for _ in atoms]
            atoms.set_initial_magnetic_moments(magmoms)

        input_data = {
            "control": {
                "calculation": "scf",
                "restart_mode": "from_scratch",
                "pseudo_dir": str(self.config.pseudopotential_dir),
                "outdir": str(run_dir),
            },
            "system": {
                "ecutwfc": self.config.ecutwfc,
                "occupations": "smearing",
                "smearing": self.config.smearing,
                "degauss": 0.02,
            },
            "electrons": {
                "conv_thr": self.config.scf_convergence_threshold,
                "mixing_beta": self.config.mixing_beta,
            },
        }

        write_pw_input(atoms, input_file, input_data, pseudos, kpts=kpts)

    def _execute_calculation(self, input_file: Path, output_file: Path, run_dir: Path) -> None:
        """Executes the pw.x command."""
        cmd_parts = self.config.command.split()
        # Standard QE usage: pw.x -in pw.in > pw.out
        cmd_parts.extend(["-in", input_file.name])

        logger.info(f"Running DFT: {' '.join(cmd_parts)} in {run_dir}")

        try:
            with output_file.open("w") as f_out:
                subprocess.run(
                    cmd_parts,
                    cwd=run_dir,
                    stdout=f_out,
                    stderr=subprocess.STDOUT,  # Redirect stderr to stdout log
                    timeout=14400,  # 4 hours
                    check=True,
                )
        except subprocess.TimeoutExpired as e:
            msg = "DFT calculation timed out."
            raise DFTRuntimeError(msg) from e
        except subprocess.CalledProcessError:
            # Check if parsing can still recover error message or detect failure
            # We proceed to parse_pw_output, which checks for "JOB DONE".
            pass
        except Exception as e:
            msg = f"Failed to execute DFT command: {e}"
            raise DFTRuntimeError(msg) from e

    def _parse_results(self, output_file: Path) -> DFTResult:
        """Parses the calculation results."""
        try:
            return parse_pw_output(output_file)
        except DFTError:
            raise
        except Exception as e:
            msg = f"Unexpected error parsing output: {e}"
            raise DFTRuntimeError(msg) from e

    def _cleanup(self, run_dir: Path) -> None:
        """Removes large temporary files."""
        # Remove bulky temporary files
        for item in run_dir.iterdir():
            if item.suffix in [".wfc", ".hub", ".mix", ".save"]:
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception:
                    logger.warning(f"Failed to cleanup file {item}")
