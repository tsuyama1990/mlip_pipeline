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
            RuntimeError: If an unexpected error occurs during masking.
        """
        try:
            # Edge case: Empty atoms
            if len(atoms) == 0:
                # Initialize empty array to ensure consistency
                atoms.new_array("force_mask", np.zeros(0, dtype=float))
                return

            positions = atoms.get_positions()
            cell = atoms.get_cell()
            pbc = atoms.pbc

            # Safe MIC calculation
            diff = positions - center

            # Only apply MIC if PBC is enabled and cell is valid
            if np.any(pbc):
                # Explicit check for cell volume to ensure it's a valid periodic box
                vol = abs(cell.volume)
                if vol < 1e-6:
                    raise ValueError(f"Atoms object has zero or near-zero cell volume ({vol}) with PBC enabled: {pbc}")

                # Check for orthogonal cell for fast path
                # L = box dimensions
                L = np.diag(cell)
                is_orthogonal = np.allclose(cell, np.diag(L))

                if is_orthogonal:
                    # Avoid division by zero if some dimension is 0
                    for i in range(3):
                        if pbc[i]:
                            # Robustness: Check for zero division
                            if abs(L[i]) < 1e-9:
                                raise ValueError(f"Cell dimension {i} is zero but PBC is enabled.")
                            diff[:, i] = diff[:, i] - np.round(diff[:, i] / L[i]) * L[i]
                else:
                     # Fallback to ASE's find_mic for non-orthogonal cells
                     from ase.geometry import find_mic
                     diff, _ = find_mic(diff, cell, pbc)

            dists = np.linalg.norm(diff, axis=1)

            # Mask: 1.0 if dist <= radius, else 0.0
            mask = np.where(dists <= radius, 1.0, 0.0)

            # Store in arrays
            atoms.set_array("force_mask", mask, dtype=float)

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to apply force mask: {e!s}") from e
