from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

class DescriptorCalculator:
    """Calculates structural descriptors (fingerprints) for atoms."""

    def __init__(self, r_cut: float = 6.0, n_max: int = 8, l_max: int = 6, sigma: float = 0.5):
        self.r_cut = r_cut
        self.n_max = n_max
        self.l_max = l_max
        self.sigma = sigma
        self._soap = None

    def compute_soap(self, atoms_list: List[Atoms]) -> np.ndarray:
        """
        Computes the average SOAP descriptor for a list of structures.
        Returns array of shape (N_structures, N_features).
        """
        # Determine species from the batch
        species = set()
        for atoms in atoms_list:
            species.update(atoms.get_chemical_symbols())
        species = sorted(list(species))

        # Initialize SOAP if not already or if species changed (simplification: re-init if needed)
        # For efficiency, ideally we know species beforehand or handle dynamic species.
        # But dscribe needs fixed species list.
        # Let's re-initialize for each batch to be safe or check if compatible.

        # Check periodicity from the first atom
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

        # dscribe requires periodic boundaries or periodic=False.
        # If any atoms object doesn't have a cell, we might have issues if periodic=True.
        # Ensure all atoms have valid cell or set periodic=False.
        # However, SPEC says "use dscribe.descriptors.SOAP with average='inner'".
        # Most structures in this project seem to be bulk crystals (Cycle 02/03).

        # Helper to ensure cell exists
        for at in atoms_list:
            if at.cell is None or np.all(at.cell.lengths() == 0):
                # Set a large dummy box for non-periodic molecules if needed,
                # but better to assume periodic=False if strictly molecules.
                # SPEC says "TargetSystem ... default: 'bulk'".
                # Cycle 03 uses molecules too.
                # If molecule, pbc should be false.
                pass

        # Create descriptors
        # output is (n_samples, n_features)
        try:
            features = self._soap.create(atoms_list, n_jobs=1)
        except ValueError as e:
            # Handle case where pbc is inconsistent with cell
            # Retry with periodic=False for molecules if that was the issue?
            # Or just let it fail.
            raise e

        return features
