import numpy as np
from ase import Atoms


class ForceMasker:
    """
    Applies force masking to an atomic structure based on distance from a center.
    """
    def apply(self, atoms: Atoms, center: np.ndarray, radius: float) -> None:
        """
        Applies force mask to atoms.

        Args:
            atoms: The Atoms object to mask (modified in place).
            center: The center position [x, y, z].
            radius: The core radius. Atoms within this radius get mask=1.0.

        Raises:
            ValueError: If atoms object has invalid cell (e.g., zero volume) when PBC is enabled.
        """
        if len(atoms) == 0:
            return

        positions = atoms.get_positions()
        cell = atoms.get_cell()
        pbc = atoms.pbc

        # Safe MIC calculation
        diff = positions - center

        # Only apply MIC if PBC is enabled and cell is valid
        if np.any(pbc):
            # Explicit check for cell volume to ensure it's a valid periodic box
            if abs(cell.volume) < 1e-6:
                raise ValueError("Atoms object has zero or near-zero cell volume with PBC enabled.")

            # Check for orthogonal cell for fast path
            # L = box dimensions
            L = np.diag(cell)
            is_orthogonal = np.allclose(cell, np.diag(L))

            if is_orthogonal:
                # Avoid division by zero if some dimension is 0 (should imply pbc=False for that dim, but be safe)
                # We only wrap dimensions where pbc is True and L > 0
                for i in range(3):
                    if pbc[i] and L[i] > 1e-6:
                        diff[:, i] = diff[:, i] - np.round(diff[:, i] / L[i]) * L[i]
            else:
                 # Fallback to ASE's find_mic for non-orthogonal cells
                 from ase.geometry import find_mic
                 # find_mic expects (N, 3) vectors and cell
                 diff, _ = find_mic(diff, cell, pbc)

        dists = np.linalg.norm(diff, axis=1)

        # Mask: 1.0 if dist <= radius, else 0.0
        mask = np.where(dists <= radius, 1.0, 0.0)

        # Store in arrays
        atoms.set_array("force_mask", mask, dtype=float)
