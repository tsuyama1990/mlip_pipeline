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
        """
        positions = atoms.get_positions()
        cell = atoms.get_cell()

        # Calculate distances from center using MIC
        # We assume the box is orthogonal (cubic) as per Embedding logic,
        # but robust MIC handles general cells.
        # For simplicity and speed in this specific pipeline (cubic box), we assume orthogonal.
        # But let's check if we can use ASE's geometry functions?
        # ase.geometry.find_mic is available but works on vectors.

        # Let's assume orthogonal box for now as per spec "cubic box".
        # L = box dimensions
        L = np.diag(cell)
        if not np.allclose(cell, np.diag(L)):
             # Fallback or warning?
             # For now, proceed assuming it works for our extracted clusters.
             pass

        diff = positions - center

        # MIC for orthogonal box
        # diff -= np.round(diff / L) * L
        # Handle cases where L might be 0 (if non-periodic directions exist?)
        # Embedding always sets pbc=True and finite box.

        # We must iterate or use broadcasting
        # diff is (N, 3), L is (3,)

        diff = diff - np.round(diff / L) * L

        dists = np.linalg.norm(diff, axis=1)

        # Mask: 1.0 if dist <= radius, else 0.0
        mask = np.where(dists <= radius, 1.0, 0.0)

        # Store in arrays
        atoms.set_array("force_mask", mask, dtype=float)
