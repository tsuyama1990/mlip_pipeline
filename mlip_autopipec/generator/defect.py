import logging
from typing import List

import numpy as np
from ase import Atoms

from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.exceptions import GeneratorError

logger = logging.getLogger(__name__)


class DefectGenerator:
    """
    Generator for point defects (vacancies and interstitials) in atomic structures.
    """

    def __init__(self, config: GeneratorConfig):
        """
        Initialize the DefectGenerator.

        Args:
            config (GeneratorConfig): The generator configuration.
        """
        self.config = config

    def create_vacancy(self, atoms: Atoms) -> List[Atoms]:
        """
        Creates vacancies by removing atoms.

        Iterates through all atomic sites and creates a structure with that atom removed.
        Currently uses a naive approach without symmetry reduction.

        Args:
            atoms (Atoms): The perfect crystal structure.

        Returns:
            List[Atoms]: A list of structures with one vacancy each.

        Raises:
            GeneratorError: If vacancy generation fails.
        """
        if not self.config.defects.enabled or not self.config.defects.vacancies:
            return []

        results = []
        n_atoms = len(atoms)

        if n_atoms <= 1:
            logger.warning("Cannot create vacancy in a structure with 1 or fewer atoms.")
            return []

        try:
            # Naive approach: remove each atom once.
            for i in range(n_atoms):
                new_atoms = atoms.copy()
                del new_atoms[i]
                new_atoms.info['config_type'] = 'vacancy'
                new_atoms.info['defect_index'] = i
                results.append(new_atoms)

            return results
        except Exception as e:
            raise GeneratorError(f"Failed to create vacancies: {e}") from e

    def create_interstitial(self, atoms: Atoms, element: str) -> List[Atoms]:
        """
        Creates interstitials by inserting an atom at heuristic void positions.

        Tries to place atoms at high-symmetry fractional coordinates (e.g., 0.5, 0.5, 0.5)
        and checks for minimum distance to existing atoms.

        Args:
            atoms (Atoms): The perfect crystal structure.
            element (str): The chemical symbol of the interstitial atom.

        Returns:
            List[Atoms]: A list of structures with one interstitial inserted.

        Raises:
            GeneratorError: If interstitial generation fails.
        """
        if not self.config.defects.enabled or not self.config.defects.interstitials:
            return []

        results = []
        try:
            candidates = [
                [0.5, 0.5, 0.5],
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.75],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5]
            ]

            cell = atoms.get_cell()

            for frac in candidates:
                pos = np.dot(frac, cell)

                # Distance check manually
                min_dist = np.inf
                for p in atoms.positions:
                    d = np.linalg.norm(p - pos)
                    if d < min_dist:
                        min_dist = d

                # Minimal distance threshold (Angstroms)
                if min_dist > 1.5:
                    new_atoms = atoms.copy()
                    new_atoms.append(element)
                    # Set position of the newly added atom (last index)
                    new_atoms.positions[-1] = pos
                    new_atoms.info['config_type'] = 'interstitial'
                    new_atoms.info['interstitial_element'] = element
                    results.append(new_atoms)

            return results
        except Exception as e:
            raise GeneratorError(f"Failed to create interstitials: {e}") from e
