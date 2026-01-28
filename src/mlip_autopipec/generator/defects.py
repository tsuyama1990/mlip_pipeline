import logging
from collections.abc import Iterable

import numpy as np
import spglib
from ase import Atoms
from ase.geometry import get_distances
from scipy.spatial import Voronoi

from mlip_autopipec.config.schemas.generator import DefectConfig
from mlip_autopipec.exceptions import GeneratorError

logger = logging.getLogger(__name__)


class DefectStrategy:
    """
    Strategy for generating point defects (vacancies and interstitials).

    This class handles the creation of defects in atomic structures.
    """

    def __init__(self, config: DefectConfig, seed: int | None = None) -> None:
        """
        Initialize the DefectStrategy.

        Args:
            config: Defect configuration.
            seed: Random seed for deterministic generation.
        """
        self.config = config
        self.rng = np.random.default_rng(seed)

    def apply(self, structures: list[Atoms], primary_element: str | None = None) -> list[Atoms]:
        """
        Applies defects to a list of structures based on configuration.

        Args:
            structures: List of input atomic structures.
            primary_element: The primary element symbol, used for interstitial fallback.

        Returns:
            list[Atoms]: A list containing the original structures and the newly generated defect structures.
        """
        if not self.config.enabled:
            return structures

        # Validate input structures
        for s in structures:
            if not isinstance(s, Atoms):
                msg = f"Input structure is not an Atoms object: {type(s)}"
                raise TypeError(msg)

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

        If `count` is 1, generates all unique single vacancy structures (exhaustive).
        If `count` > 1, generates a single structure with `count` randomly removed atoms.

        Args:
            atoms: The input structure.
            count: Number of vacancies to create per structure.

        Returns:
            list[Atoms]: List of structures with vacancies.
        """
        if not isinstance(atoms, Atoms):
            msg = f"Input structure is not an Atoms object: {type(atoms)}"
            raise TypeError(msg)

        results = []
        n_atoms = len(atoms)
        if n_atoms <= count:
            return []

        try:
            # If structure is small, we can exhaustively generate all single vacancies
            if count == 1:
                # Use spglib to find symmetry-equivalent atoms for efficiency
                # We need cell as (lattice, positions, numbers)
                cell = (atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
                dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)

                indices_to_remove: Iterable[int]
                if dataset is None:
                    # Fallback if symmetry analysis fails
                    indices_to_remove = range(n_atoms)
                else:
                    equivalent_atoms = dataset["equivalent_atoms"]
                    # Find unique representatives
                    _, unique_indices = np.unique(equivalent_atoms, return_index=True)
                    indices_to_remove = sorted(unique_indices)

                # Generate vacancies for unique sites
                for i in indices_to_remove:
                    new_atoms = atoms.copy()
                    del new_atoms[i]
                    new_atoms.info["config_type"] = "vacancy"
                    new_atoms.info["defect_index"] = int(i)
                    results.append(new_atoms)
            else:
                # Randomly remove 'count' atoms once
                # Use self.rng for determinism
                indices = self.rng.choice(n_atoms, size=count, replace=False).tolist()
                new_atoms = atoms.copy()
                # Delete in reverse order to preserve indices
                for i in sorted(indices, reverse=True):
                    del new_atoms[i]
                new_atoms.info["config_type"] = "vacancy"
                new_atoms.info["defect_indices"] = indices
                results.append(new_atoms)

        except Exception as e:
            msg = f"Vacancy generation failed: {e}"
            logger.error(msg, exc_info=True)
            raise GeneratorError(msg) from e

        return results

    def generate_interstitials(self, atoms: Atoms, element: str = "H") -> list[Atoms]:
        """
        Generates structures with one interstitial atom.

        Uses Voronoi tessellation to identify void spaces. If the structure is too small
        for Voronoi (fewer than 4 atoms), falls back to heuristic fractional coordinates.

        Args:
            atoms: The input structure.
            element: The chemical symbol of the interstitial atom.

        Returns:
            list[Atoms]: List of structures with an interstitial atom inserted.
        """
        if not isinstance(atoms, Atoms):
            msg = f"Input structure is not an Atoms object: {type(atoms)}"
            raise TypeError(msg)

        results = []
        try:
            # Let's try Voronoi if enough points, otherwise fallback
            if len(atoms) < 4:
                # Fallback to simple fractional coordinates
                candidates = [
                    np.array([0.5, 0.5, 0.5]),
                    np.array([0.25, 0.25, 0.25]),
                    np.array([0.75, 0.75, 0.75]),
                    np.array([0.5, 0.0, 0.0]),
                    np.array([0.0, 0.5, 0.0]),
                    np.array([0.0, 0.0, 0.5]),
                    np.array([0.5, 0.25, 0.0]),
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
                    scaled = atoms.cell.scaled_positions(vert.reshape(1, 3))
                    # wrap
                    scaled = scaled % 1.0
                    candidates.append(scaled.flatten())

            # Filter candidates
            unique_candidates: list[np.ndarray] = []
            min_dist = self.config.interstitial_min_dist
            cluster_cutoff = self.config.interstitial_cluster_cutoff

            for c in candidates:
                # Check distance to existing atoms
                pos = np.dot(c, atoms.get_cell())

                D_vectors, D_scalar = get_distances(
                    atoms.positions, pos.reshape(1, 3), cell=atoms.cell, pbc=atoms.pbc
                )
                dists = D_scalar.flatten()

                # Check min distance and uniqueness against other candidates
                if np.min(dists) > min_dist and not any(
                    np.linalg.norm(uc - c) < cluster_cutoff for uc in unique_candidates
                ):
                    unique_candidates.append(c)

            # Limit number of interstitials per structure to avoid explosion
            for c in unique_candidates[:5]:
                new_atoms = atoms.copy()
                new_atoms.append(element)
                new_atoms.positions[-1] = np.dot(c, atoms.get_cell())
                new_atoms.info["config_type"] = "interstitial"
                new_atoms.info["interstitial_element"] = element
                results.append(new_atoms)

        except Exception as e:
            msg = f"Interstitial generation failed: {e}"
            logger.error(msg, exc_info=True)
            raise GeneratorError(msg) from e

        return results
