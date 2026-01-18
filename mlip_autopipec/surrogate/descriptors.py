from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

class DescriptorCalculator:
    """
    Calculates structural descriptors (fingerprints) for atoms using DScribe.
    """

    def __init__(self, r_cut: float = 6.0, n_max: int = 8, l_max: int = 6, sigma: float = 0.5):
        """
        Initialize the DescriptorCalculator.

        Args:
            r_cut: Cutoff radius in Angstroms.
            n_max: Number of radial basis functions.
            l_max: Maximum degree of spherical harmonics.
            sigma: Standard deviation of the gaussians.
        """
        self.r_cut = r_cut
        self.n_max = n_max
        self.l_max = l_max
        self.sigma = sigma
        self._soap = None

    def compute_soap(self, atoms_list: List[Atoms]) -> np.ndarray:
        """
        Computes the average SOAP descriptor for a list of structures.

        Args:
            atoms_list: List of ASE Atoms objects.

        Returns:
            np.ndarray: Array of shape (N_structures, N_features).

        Raises:
            ValueError: If SOAP calculation fails.
        """
        if not atoms_list:
            return np.array([])

        # Determine species from the batch
        species = set()
        for atoms in atoms_list:
            species.update(atoms.get_chemical_symbols())
        species = sorted(list(species))

        # Check periodicity from the first atom
        # We assume homogeneity in batch for periodicity (all bulk or all molecules)
        # If mixed, we might need to handle differently, but usually it's one cycle type.
        is_periodic = False
        if atoms_list and hasattr(atoms_list[0], 'pbc'):
            is_periodic = np.any(atoms_list[0].pbc)

        self._soap = SOAP(
            species=species,
            periodic=is_periodic,
            r_cut=self.r_cut,
            n_max=self.n_max,
            l_max=self.l_max,
            sigma=self.sigma,
            average="inner", # Average over atoms to get global descriptor
            sparse=False
        )

        try:
            # dscribe creates features
            features = self._soap.create(atoms_list, n_jobs=1)
        except Exception as e:
            # Catch dscribe errors
            raise ValueError(f"Failed to compute SOAP descriptors: {e}")

        return features
