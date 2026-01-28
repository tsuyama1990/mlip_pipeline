"""
Module for parsing Quantum Espresso output files.
"""

from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read as ase_read

from mlip_autopipec.domain_models.dft_models import DFTResult


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
            Exception: If the output file cannot be parsed or indicates failure.
        """
        # 1. robustly verify completion
        try:
            content = output_path.read_text(errors="replace")
            if "JOB DONE" not in content:
                msg = "QE output missing 'JOB DONE' marker."
                raise Exception(msg)
        except FileNotFoundError:
            msg = f"Output file not found: {output_path}"
            raise Exception(msg)

        try:
            # ase.io.read returns Atoms object
            # format='espresso-out' is auto-detected usually
            # We use 'espresso-out' specifically to avoid ambiguity
            atoms_out = self.reader(output_path, format="espresso-out")

            energy = atoms_out.get_potential_energy()

            forces = atoms_out.get_forces()

            # Check for NaN/Inf in forces
            if np.isnan(forces).any() or np.isinf(forces).any():
                msg = "Forces contain NaN or Inf values."
                raise ValueError(msg)

            stress = atoms_out.get_stress(voigt=False)  # Get 3x3 tensor

            # Check for NaN/Inf in stress
            if np.isnan(stress).any() or np.isinf(stress).any():
                msg = "Stress contains NaN or Inf values."
                raise ValueError(msg)

            if hasattr(forces, "tolist"):
                forces = forces.tolist()
            if hasattr(stress, "tolist"):
                stress = stress.tolist()

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
            msg = f"Failed to parse output: {e}"
            raise Exception(msg) from e
