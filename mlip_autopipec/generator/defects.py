import logging
import random

import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from scipy.spatial import Voronoi

from mlip_autopipec.config.schemas.generator import DefectConfig
from mlip_autopipec.exceptions import GeneratorError

logger = logging.getLogger(__name__)


class DefectStrategy:
    """
    Strategy for generating point defects (vacancies and interstitials).
    """

    def __init__(self, config: DefectConfig) -> None:
        self.config = config

    def apply(self, structures: list[Atoms], primary_element: str | None = None) -> list[Atoms]:
        """
        Applies defects to a list of structures based on configuration.
        """
        if not self.config.enabled:
            return structures

        new_structures = list(structures)

        # Determine interstitial elements
        elements_to_insert = self.config.interstitial_elements
        if not elements_to_insert and primary_element:
            elements_to_insert = [primary_element]

        for s in structures:
            if self.config.vacancies:
                vac_structures = self.generate_vacancies(s, count=1)
                new_structures.extend(vac_structures)

            if self.config.interstitials and elements_to_insert:
                for el in elements_to_insert:
                    int_structures = self.generate_interstitials(s, el)
                    new_structures.extend(int_structures)

        return new_structures

    def generate_vacancies(self, atoms: Atoms, count: int = 1) -> list[Atoms]:
        """
        Generates structures with vacancies.
        Currently generates all unique single vacancies (if count=1).
        If count > 1, random sampling might be better.
        """
        results = []
        n_atoms = len(atoms)
        if n_atoms <= count:
            return []

        try:
            # If structure is small, we can exhaustively generate all single vacancies
            if count == 1:
                # Generate all single vacancies
                for i in range(n_atoms):
                    new_atoms = atoms.copy() # type: ignore[no-untyped-call]
                    del new_atoms[i]
                    new_atoms.info["config_type"] = "vacancy"
                    new_atoms.info["defect_index"] = i
                    results.append(new_atoms)
            else:
                 # Randomly remove 'count' atoms once
                 indices = random.sample(range(n_atoms), count)
                 new_atoms = atoms.copy() # type: ignore[no-untyped-call]
                 # Delete in reverse order to preserve indices
                 for i in sorted(indices, reverse=True):
                     del new_atoms[i]
                 new_atoms.info["config_type"] = "vacancy"
                 new_atoms.info["defect_indices"] = indices
                 results.append(new_atoms)

        except Exception as e:
            msg = f"Vacancy generation failed: {e}"
            raise GeneratorError(msg) from e

        return results

    def generate_interstitials(self, atoms: Atoms, element: str = "H") -> list[Atoms]:
        """
        Generates structures with one interstitial atom using Voronoi tessellation.
        """
        results = []
        try:
            # Let's try Voronoi if enough points, otherwise fallback
            # Ensure we use arrays for coordinates to avoid subtraction errors later
            if len(atoms) < 4:
                 # Fallback to simple fractional coordinates
                 # Try multiple potential sites including tetrahedral voids
                 candidates = [
                     np.array([0.5, 0.5, 0.5]),
                     np.array([0.25, 0.25, 0.25]),
                     np.array([0.75, 0.75, 0.75]),
                     np.array([0.5, 0.0, 0.0]),
                     np.array([0.0, 0.5, 0.0]),
                     np.array([0.0, 0.0, 0.5]),
                     np.array([0.5, 0.25, 0.0]), # Tetrahedral
                     np.array([0.5, 0.75, 0.0]),
                     np.array([0.0, 0.5, 0.25]),
                     np.array([0.25, 0.0, 0.5]),
                 ]
            else:
                 # Voronoi
                 v = Voronoi(atoms.positions)
                 candidates = []
                 for vert in v.vertices:
                     # Map back to cell
                     scaled = atoms.cell.scaled_positions(vert.reshape(1,3)) # type: ignore
                     # wrap
                     scaled = scaled % 1.0
                     candidates.append(scaled.flatten())

            # Filter candidates
            unique_candidates = []
            for c in candidates:
                # Check distance to existing atoms
                pos = np.dot(c, atoms.get_cell())

                # Calculate distances with MIC using ase.geometry.get_distances
                # get_distances(p1, p2, cell, pbc) returns (vectors, distances)
                D_vectors, D_scalar = get_distances(atoms.positions, pos.reshape(1, 3), cell=atoms.cell, pbc=atoms.pbc)
                dists = D_scalar.flatten()

                # Check min distance > 1.4A
                if np.min(dists) > 1.4:
                     # Check uniqueness against other candidates
                     if not any(np.linalg.norm(uc - c) < 0.1 for uc in unique_candidates):
                          unique_candidates.append(c)

            # Limit number of interstitials per structure to avoid explosion
            for c in unique_candidates[:5]:
                new_atoms = atoms.copy() # type: ignore[no-untyped-call]
                new_atoms.append(element)
                new_atoms.positions[-1] = np.dot(c, atoms.get_cell())
                new_atoms.info["config_type"] = "interstitial"
                new_atoms.info["interstitial_element"] = element
                results.append(new_atoms)

        except Exception as e:
             msg = f"Interstitial generation failed: {e}"
             raise GeneratorError(msg) from e

        return results
