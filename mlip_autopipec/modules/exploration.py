"""
This module contains the SurrogateExplorer class, responsible for intelligent
pre-screening and selection of atomic structures before they are sent for
expensive DFT calculations.
"""


import numpy as np
import torch
from ase import Atoms
from dscribe.descriptors import SOAP
from mace.calculators import mace_mp

from mlip_autopipec.config.models import ExplorerConfig


class SurrogateExplorer:
    """
    Uses a surrogate model (MACE) and Farthest Point Sampling (FPS) to
    curate a list of candidate structures.
    """

    def __init__(self, config: ExplorerConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mace_model = mace_mp(
            model=self.config.surrogate_model_path, device=self.device
        )
        self.fingerprint_generator = self._init_fingerprint_generator()

    def _init_fingerprint_generator(self):
        """Initializes the fingerprint generator based on the config."""
        fp_config = self.config.fingerprint
        if fp_config.type == "soap":
            return SOAP(
                species=fp_config.species,
                r_cut=fp_config.soap_rcut,
                n_max=fp_config.soap_nmax,
                l_max=fp_config.soap_lmax,
                periodic=True,
                sparse=False,
                average="inner",
            )
        # This can be extended to support other fingerprint types
        raise NotImplementedError(f"Fingerprint type '{fp_config.type}' not supported.")

    def select(self, structures: list[Atoms], num_to_select: int) -> list[Atoms]:
        """
        Orchestrates the full selection workflow: pre-screening, fingerprinting,
        and Farthest Point Sampling.

        Args:
            structures: A list of candidate ase.Atoms objects.
            num_to_select: The target number of structures to select.

        Returns:
            A smaller, curated list of ase.Atoms objects.
        """
        screened_structures = self._pre_screen_structures(structures)

        if not screened_structures or len(screened_structures) <= num_to_select:
            return screened_structures

        fingerprints = self._calculate_fingerprints(screened_structures)
        selected_indices = self._farthest_point_sampling(fingerprints, num_to_select)

        final_selection = [screened_structures[i] for i in selected_indices]
        return final_selection

    def _pre_screen_structures(self, structures: list[Atoms]) -> list[Atoms]:
        """
        Filters out unstable structures using the MACE surrogate model.
        """
        passed_structures = []
        for atoms in structures:
            atoms.calc = self.mace_model
            forces = atoms.get_forces()
            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force < self.config.max_force_threshold:
                passed_structures.append(atoms)
        return passed_structures

    def _calculate_fingerprints(self, structures: list[Atoms]) -> np.ndarray:
        """
        Computes structural fingerprints for a list of structures.
        """
        return self.fingerprint_generator.create(structures, n_jobs=-1)

    @staticmethod
    def _farthest_point_sampling(points: np.ndarray, num_to_select: int) -> list[int]:
        """
        Selects a diverse subset of points using the Farthest Point Sampling algorithm.
        """
        n_points = points.shape[0]
        if n_points <= num_to_select:
            return list(range(n_points))

        selected_indices = np.zeros(num_to_select, dtype=int)
        min_dist_sq = np.full(n_points, np.inf)

        # Start with the first point
        selected_indices[0] = 0

        for i in range(1, num_to_select):
            last_selected_idx = selected_indices[i - 1]
            last_selected_pt = points[last_selected_idx]

            # Calculate distance from all points to the last selected point
            dist_sq_to_last = np.sum((points - last_selected_pt) ** 2, axis=1)

            # Update the minimum distance for each point
            min_dist_sq = np.minimum(min_dist_sq, dist_sq_to_last)

            # Select the point with the maximum minimum distance
            selected_indices[i] = np.argmax(min_dist_sq)

        return selected_indices.tolist()
