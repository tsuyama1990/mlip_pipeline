"""Module for parsing Quantum Espresso output files."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError

from mlip_autopipec.modules.dft.exceptions import DFTCalculationError

logger = logging.getLogger(__name__)


class QEOutputParser:
    """Parses Quantum Espresso output files."""

    def parse(self, output_path: Path) -> dict[str, Any]:
        """Parse the output file from Quantum Espresso to extract results.

        Args:
            output_path: Path to the QE output file.

        Returns:
            A dictionary containing the parsed energy, forces, and stress.

        """
        from ase.io.espresso import read_espresso_out

        with open(output_path) as f:
            # ASE's parser is robust and well-tested
            parsed_atoms_list = list(read_espresso_out(f, index=slice(None)))

        if not parsed_atoms_list:
            raise DFTCalculationError(
                "Failed to parse any configuration from QE output."
            )

        final_atoms = parsed_atoms_list[-1]
        try:
            stress = final_atoms.get_stress(voigt=False)
        except PropertyNotImplementedError:
            logger.warning("Stress tensor not found in QE output. Setting to zeros.")
            stress = np.zeros((3, 3))

        return {
            "energy": final_atoms.get_potential_energy(),
            "forces": final_atoms.get_forces(),
            "stress": stress,
        }
