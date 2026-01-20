"""
Module for parsing Quantum Espresso output files.
"""

from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read as ase_read

from mlip_autopipec.data_models.dft_models import DFTResult
from mlip_autopipec.exceptions import DFTException


def parse_pw_output(
    output_path: Path,
    uid: str,
    wall_time: float,
    parameters: dict[str, Any],
) -> DFTResult:
    """
    Parses the `espresso.pwo` output file of a successful QE run.
    """
    if not output_path.exists():
        msg = f"Output file {output_path} does not exist."
        raise DFTException(msg)

    content = output_path.read_text()
    if "JOB DONE" not in content:
        msg = "DFT calculation did not complete successfully (JOB DONE not found)."
        raise DFTException(msg)

    try:
        atoms_out = ase_read(output_path, format="espresso-out")
    except Exception as e:
        msg = f"Failed to parse output: {e}"
        raise DFTException(msg) from e

    energy = atoms_out.get_potential_energy()
    forces = atoms_out.get_forces()

    # stress in ASE can be Voigt (6 elements) or 3x3
    # Spec wants 3x3.
    stress = atoms_out.get_stress(voigt=False)

    # Sanity check
    if np.isnan(forces).any() or np.isinf(forces).any():
        msg = "Forces contain NaN or Inf."
        raise DFTException(msg)

    if hasattr(forces, "tolist"):
        forces = forces.tolist()
    if hasattr(stress, "tolist"):
        stress = stress.tolist()

    final_beta = parameters.get("mixing_beta", 0.7)

    return DFTResult(
        uid=uid,
        energy=energy,
        forces=forces,
        stress=stress,
        succeeded=True,
        wall_time=wall_time,
        parameters=parameters,
        final_mixing_beta=final_beta,
    )


class QEOutputParser:
    """
    Parses a Quantum Espresso output file into a `DFTResult` object.
    """

    def __init__(self, reader: Any = ase_read) -> None:
        """
        Initializes the QEOutputParser.

        Args:
            reader: A callable (like `ase.io.read`) that can parse QE output
                    files. This is dependency-injected for testability.
        """
        self.reader = reader

    def parse(
        self,
        output_path: Path,
        uid: str,
        wall_time: float,
        params: dict[str, Any],
    ) -> DFTResult:
        """
        Parses the `espresso.pwo` output file of a successful QE run.

        Args:
            output_path: The path to the QE output file.
            uid: The unique identifier for the DFT job.
            wall_time: Execution time in seconds.
            params: Parameters used for the calculation.

        Returns:
            A `DFTResult` object containing the parsed energy, forces, and
            stress.

        Raises:
            Exception: If the output file cannot be parsed.
        """
        try:
            # ase.io.read returns Atoms object
            # format='espresso-out' is auto-detected usually
            atoms_out = self.reader(output_path, format="espresso-out")

            energy = atoms_out.get_potential_energy()

            # Helper to safely convert numpy arrays to lists if needed
            forces = atoms_out.get_forces()
            if hasattr(forces, "tolist"):
                forces = forces.tolist()

            stress = atoms_out.get_stress(voigt=False)  # Get 3x3 tensor
            if hasattr(stress, "tolist"):
                stress = stress.tolist()

            # Helper to get mixing beta if available?
            # ASE might not parse it easily. We use the one from params or default
            final_beta = params.get("mixing_beta", 0.7)

            return DFTResult(
                uid=uid,
                energy=energy,
                forces=forces,
                stress=stress,
                succeeded=True,
                wall_time=wall_time,
                parameters=params,
                final_mixing_beta=final_beta,
            )
        except Exception as e:
            # If ASE fails, it means calculation didn't finish or crashed
            raise Exception(f"Failed to parse output: {e}") from e
