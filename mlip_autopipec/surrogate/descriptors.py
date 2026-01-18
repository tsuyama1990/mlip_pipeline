import logging
from typing import List, Optional
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP
from mlip_autopipec.config.schemas.surrogate import DescriptorConfig

logger = logging.getLogger(__name__)

class DescriptorCalculator:
    """
    Calculates structural descriptors (fingerprints) for atoms using DScribe.

    This class wraps DScribe functionalities (currently SOAP) to provide a
    standardized interface for featurizing atomic structures.
    """

    def __init__(self, config: DescriptorConfig):
        """
        Initialize the DescriptorCalculator.

        Args:
            config: A DescriptorConfig object containing parameters like r_cut, n_max, etc.
        """
        self.config = config
        self._soap: Optional[SOAP] = None

    def compute_soap(self, atoms_list: List[Atoms]) -> np.ndarray:
        """
        Computes the average SOAP descriptor for a list of structures.

        Args:
            atoms_list: List of ASE Atoms objects to featurize.

        Returns:
            np.ndarray: Array of shape (N_structures, N_features).

        Raises:
            ValueError: If the atoms list is invalid or empty when processing logic requires valid input.
            RuntimeError: If the internal descriptor calculation fails.
        """
        if not atoms_list:
            logger.warning("Empty atoms list provided to compute_soap.")
            return np.array([])

        # Determine species from the batch
        species = set()
        for atoms in atoms_list:
            species.update(atoms.get_chemical_symbols())
        species = sorted(list(species))

        # Check periodicity from the first atom
        is_periodic = False
        if atoms_list and hasattr(atoms_list[0], 'pbc'):
            is_periodic = np.any(atoms_list[0].pbc)

        try:
            self._soap = SOAP(
                species=species,
                periodic=is_periodic,
                r_cut=self.config.r_cut,
                n_max=self.config.n_max,
                l_max=self.config.l_max,
                sigma=self.config.sigma,
                average="inner", # Average over atoms to get global descriptor
                sparse=False
            )

            # dscribe create method
            features = self._soap.create(atoms_list, n_jobs=1)

        except Exception as e:
            logger.error(f"SOAP calculation failed: {e}")
            raise RuntimeError(f"Failed to compute SOAP descriptors: {e}") from e

        return features
