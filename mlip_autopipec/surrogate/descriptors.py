import logging

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

from mlip_autopipec.config.schemas.surrogate import DescriptorConfig, DescriptorResult

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
        self._soap: SOAP | None = None

    def compute_soap(self, atoms_list: list[Atoms]) -> DescriptorResult:
        """
        Computes the average SOAP descriptor for a list of structures.

        Args:
            atoms_list: List of ASE Atoms objects to featurize.

        Returns:
            DescriptorResult: Object containing the features array.

        Raises:
            ValueError: If the atoms list is invalid or empty when processing logic requires valid input.
            RuntimeError: If the internal descriptor calculation fails.
        """
        if not atoms_list:
            logger.warning("Empty atoms list provided to compute_soap.")
            # Return empty 2D array
            return DescriptorResult(features=np.empty((0, 0)))

        # Determine species from the batch
        species = set()
        for atoms in atoms_list:
            species.update(atoms.get_chemical_symbols())
        species = sorted(list(species))

        # Check periodicity from the first atom
        is_periodic = False
        if atoms_list and hasattr(atoms_list[0], "pbc"):
            is_periodic = np.any(atoms_list[0].pbc)

        try:
            self._soap = SOAP(
                species=species,
                periodic=is_periodic,
                r_cut=self.config.r_cut,
                n_max=self.config.n_max,
                l_max=self.config.l_max,
                sigma=self.config.sigma,
                average="inner",  # Average over atoms to get global descriptor
                sparse=False,
            )

            # dscribe create method
            features = self._soap.create(atoms_list, n_jobs=1)

            # Ensure it's 2D (dscribe might return 1D if single sample and flat output, though usually it respects batch)
            # If features is 1D, reshape to (1, -1)
            if features.ndim == 1:
                features = features.reshape(1, -1)

            return DescriptorResult(features=features)

        except Exception as e:
            logger.error(f"SOAP calculation failed: {e}")
            raise RuntimeError(f"Failed to compute SOAP descriptors: {e}") from e
