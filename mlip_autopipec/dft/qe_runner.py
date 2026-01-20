"""
Quantum Espresso Runner.
"""
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ase import Atoms

from mlip_autopipec.core.config import DFTConfig
from mlip_autopipec.core.exceptions import DFTRuntimeError
from mlip_autopipec.core.models import DFTResult
from mlip_autopipec.dft import input_gen, parsers, utils


class QERunner:
    """
    Executes Quantum Espresso calculations.
    """

    def __init__(self, config: DFTConfig) -> None:
        """
        Initialize the runner.

        Args:
            config: DFT configuration.
        """
        self.config = config

    def run_static_calculation(self, atoms: Atoms, run_dir: Path) -> DFTResult:
        """
        Runs a static SCF calculation.

        Args:
            atoms: The atomic structure.
            run_dir: Directory to run the calculation in.

        Returns:
            DFTResult: The calculation results.
        """
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Sanitize / Setup
        # Ensure atoms fit in box? (Sanity check)
        # For now assume valid atoms.

        elements = list(set(atoms.get_chemical_symbols()))

        # Pseudopotentials
        # Map element -> filename
        pseudos = utils.get_sssp_pseudopotentials(elements)

        # K-Points
        kpts = utils.get_kpoints(atoms, self.config.kpoints_density)

        # Parameters
        parameters: dict[str, Any] = {
            "control": {
                "calculation": "scf",
                "pseudo_dir": str(self.config.pseudopotential_dir),
                "restart_mode": "from_scratch",
            },
            "system": {
                "ecutwfc": 60.0, # Hardcoded default or config?
                "smearing": self.config.smearing,
                "degauss": 0.02, # Default smearing width
            },
            "electrons": {
                "conv_thr": self.config.scf_convergence_threshold,
                "mixing_beta": self.config.mixing_beta,
            }
        }

        # Magnetism
        if utils.is_magnetic(atoms):
            parameters["system"]["nspin"] = 2
            parameters["system"]["starting_magnetization"] = {
                el: 0.1 for el in elements if el in ["Fe", "Ni", "Co"]
            }
            # Also set initial moments on atoms for consistency?
            # input_gen uses ase.io.write.
            # If we set parameters['system']['starting_magnetization'], ASE writes it to input.
            # But ASE might also look at atoms.get_initial_magnetic_moments().
            # I will set it in parameters to be safe as per Spec "Input Writing".

        # 2. Generate Input
        input_file = run_dir / "pw.in"
        output_file = run_dir / "pw.out"

        input_gen.write_pw_input(
            atoms=atoms,
            parameters=parameters,
            pseudopotentials=pseudos,
            kpts=kpts,
            output_path=input_file
        )

        # 3. Execute
        # Command construction
        try:
            with output_file.open("w") as f_out:
                cmd_parts = [*self.config.command.split(), "-in", "pw.in"]

                subprocess.run( # noqa: S603
                    cmd_parts,
                    cwd=run_dir,
                    stdout=f_out,
                    stderr=subprocess.STDOUT,
                    check=True, # Raises CalledProcessError if return code != 0
                    timeout=14400 # 4 hours
                )
        except subprocess.TimeoutExpired as e:
            msg = "DFT calculation timed out."
            raise DFTRuntimeError(msg) from e
        except subprocess.CalledProcessError as e:
            msg = f"DFT execution failed with return code {e.returncode}."
            raise DFTRuntimeError(msg) from e
        except Exception as e:
            msg = f"Unexpected error executing DFT: {e}"
            raise DFTRuntimeError(msg) from e

        # 4. Parse Output
        result = parsers.parse_pw_output(output_file)

        # 5. Clean
        # Remove bulky files
        # .wfc, .hub, .mix usually in prefix.save or just in dir?
        # By default prefix='ase' or similar.
        # ASE writes `prefix='calc'`? No, usually 'pwscf' default.
        # We didn't specify prefix in parameters.
        # Let's clean everything except .in and .out
        for p in run_dir.iterdir():
            if p.name not in ["pw.in", "pw.out"]:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()

        return result
