import logging
from typing import Any

import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from mlip_autopipec.config.schemas.validation import PhononConfig

logger = logging.getLogger(__name__)


class PhononValidator:
    """
    Validates the dynamic stability of a structure using Phonopy.
    Checks for imaginary frequencies in the phonon dispersion.
    """

    def __init__(self, config: PhononConfig) -> None:
        self.config = config

    def validate(self, atoms: Atoms, calculator: Any) -> bool:
        """
        Runs phonon calculation and checks stability.

        Args:
            atoms: The unit cell structure.
            calculator: ASE calculator (must implement get_forces).

        Returns:
            bool: True if stable (no imaginary modes), False otherwise.
        """
        # Convert ASE Atoms to PhonopyAtoms
        unitcell = PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            cell=atoms.get_cell(),
            scaled_positions=atoms.get_scaled_positions(),
        )

        phonon = Phonopy(
            unitcell, supercell_matrix=self.config.supercell_matrix, primitive_matrix="auto"
        )  # Let phonopy find primitive

        phonon.generate_displacements(distance=self.config.displacement)
        supercells = phonon.supercells_with_displacements

        if supercells is None:
            logger.error("Failed to generate supercells.")
            return False

        logger.info(f"Calculating forces for {len(supercells)} supercells...")

        sets_of_forces = []
        for sc in supercells:
            # Convert PhonopyAtoms back to ASE Atoms
            ase_sc = Atoms(
                symbols=sc.get_chemical_symbols(),
                positions=sc.get_positions(),
                cell=sc.get_cell(),
                pbc=True,
            )

            # Attach calculator
            # We clone the calculator if possible, or just reuse if it's stateless enough
            # But usually we just set .calc.
            # Note: If calculator is attached to original 'atoms', we should probably use that instance?
            # The 'calculator' arg is passed explicitly.
            ase_sc.calc = calculator

            # Use try-except block for calculation safety
            try:
                forces = ase_sc.get_forces()
            except Exception as e:
                logger.error(f"Force calculation failed: {e}")
                return False

            sets_of_forces.append(forces)

        phonon.produce_force_constants(forces=sets_of_forces)

        # Run Mesh calculation to sample Brillouin zone
        phonon.run_mesh(self.config.mesh)
        mesh_dict = phonon.get_mesh_dict()
        frequencies = mesh_dict["frequencies"]

        # Check for imaginary frequencies (negative values)
        # We apply a small tolerance (-0.1 THz) to account for numerical noise at Gamma
        min_freq = np.min(frequencies)
        logger.info(f"Phonon Min Frequency: {min_freq:.4f} THz")

        if min_freq < -0.1:
            logger.warning(f"Structure is dynamically unstable! Min freq: {min_freq:.4f} THz")
            return False

        logger.info("Structure is dynamically stable.")
        return True
