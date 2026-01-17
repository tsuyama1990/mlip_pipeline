"""
This module contains the `QEOutputParser` class.
"""

from pathlib import Path
from typing import Any

from ase.io import read as ase_read

from mlip_autopipec.config.models import DFTResult
from mlip_autopipec.exceptions import DFTCalculationError


class QEOutputParser:
    """
    Parses a Quantum Espresso output file into a `DFTResult` object.

    This class uses the ASE `read` function with the `espresso-out` format to
    extract the final energy, forces, and stress from a `pw.x` output file.
    It then wraps this data in a validated `DFTResult` Pydantic model.
    """

    def __init__(self, reader: Any = ase_read) -> None:
        """
        Initializes the QEOutputParser.

        Args:
            reader: A callable (like `ase.io.read`) that can parse QE output
                    files. This is dependency-injected for testability.
        """
        self.reader = reader

    def parse(self, output_path: Path, job_id: Any) -> DFTResult:
        """
        Parses the `espresso.pwo` output file of a successful QE run.

        Args:
            output_path: The path to the QE output file.
            job_id: The unique identifier for the DFT job.

        Returns:
            A `DFTResult` object containing the parsed energy, forces, and
            stress.

        Raises:
            DFTCalculationError: If the output file cannot be parsed.
        """
        try:
            result_atoms = self.reader(output_path, format="espresso-out")
            energy = result_atoms.get_potential_energy()

            # Check if forces/stress are numpy arrays or lists before tolist()
            # ASE generally returns numpy arrays
            forces = result_atoms.get_forces()
            if hasattr(forces, "tolist"):
                forces = forces.tolist()

            stress = result_atoms.get_stress(voigt=False)
            if hasattr(stress, "tolist"):
                stress = stress.tolist()

            return DFTResult(
                uid=str(job_id),
                energy=energy,
                forces=forces,
                stress=stress,
                succeeded=True,
                wall_time=0.0,  # Placeholder as this legacy parser didn't track it
                parameters={},  # Placeholder
                final_mixing_beta=0.7,  # Placeholder
            )
        except (OSError, IndexError) as e:
            msg = f"Failed to parse QE output file: {output_path}"
            raise DFTCalculationError(msg) from e
