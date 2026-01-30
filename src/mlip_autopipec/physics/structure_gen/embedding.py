from typing import Optional

import numpy as np
import ase
from scipy.spatial import cKDTree  # type: ignore

from mlip_autopipec.domain_models.structure import Structure


class EmbeddingHandler:
    """
    Handles extraction of clusters from larger structures and embedding them
    into vacuum-padded boxes for DFT calculations.
    """

    def __init__(self, radius: float = 6.0, vacuum: float = 5.0):
        self.radius = radius
        self.vacuum = vacuum

    def embed(self, structure: Structure, center_index: int) -> Structure:
        """
        Extract a cluster centered at `center_index` with `self.radius`,
        and place it in a new cubic cell with `self.vacuum` padding.
        """
        atoms = structure.to_ase()

        # Ensure we handle PBC correctly.
        # We want all atoms within radius of the center atom (considering PBC).
        # cKDTree with boxsize works well for orthogonal cells.
        # For general cells, using ASE's neighbor list is safer, but we are instructed
        # to use cKDTree.
        # If the cell is orthogonal, we pass boxsize.

        cell = atoms.get_cell() # type: ignore[no-untyped-call]
        pbc = atoms.get_pbc() # type: ignore[no-untyped-call]

        is_orthogonal = np.allclose(cell, np.diag(np.diag(cell)))
        boxsize = np.diag(cell) if (is_orthogonal and np.all(pbc)) else None

        positions = atoms.get_positions() # type: ignore[no-untyped-call]
        center_pos = positions[center_index]

        tree = cKDTree(positions, boxsize=boxsize)

        # Query indices
        indices = tree.query_ball_point(center_pos, r=self.radius)

        # Now we have indices of atoms in the cluster.
        # We need to extract them and unwrap them so they form a continuous cluster
        # relative to the center.
        # cKDTree query_ball_point returns indices, but not the image vectors.
        # To get the correct positions relative to center, we need to compute distances again
        # or use the periodic vector.

        # Easier approach with ASE to get unwrapped positions:
        # 1. Create a cluster object
        # 2. For each atom in indices, find the minimum image vector to center_pos

        cluster_positions = []
        cluster_symbols = []

        symbols = atoms.get_chemical_symbols() # type: ignore[no-untyped-call]

        # We iterate and fix PBC manually to be safe or rely on MIC
        for idx in indices:
            pos = positions[idx]
            vec = pos - center_pos
            if boxsize is not None:
                # Apply MIC for orthogonal
                vec = vec - boxsize * np.round(vec / boxsize)
            elif np.any(pbc):
                 # Fallback for non-orthogonal or partial PBC: use ASE
                 # This is slower but correct
                 d_vec = atoms.get_distance(center_index, idx, mic=True, vector=True) # type: ignore[no-untyped-call]
                 vec = d_vec

            cluster_positions.append(vec) # Relative to center
            cluster_symbols.append(symbols[idx])

        cluster_positions = np.array(cluster_positions)

        # Now shift so center is at (0,0,0) -> It is already (0,0,0) for center_index
        # We want to center it in the new box.

        # New box size
        # Diameter is 2*radius. Add vacuum.
        # box_side = 2 * (self.radius + self.vacuum)?
        # Usually vacuum is added to the bounding box of the cluster.
        # Let's say box_side = 2*radius + 2*vacuum to be safe.

        box_side = 2.0 * (self.radius + self.vacuum)
        new_cell = np.eye(3) * box_side

        # Shift positions to center of new box
        center_of_box = np.array([box_side/2, box_side/2, box_side/2])
        new_positions = cluster_positions + center_of_box

        new_atoms = ase.Atoms(
            symbols=cluster_symbols,
            positions=new_positions,
            cell=new_cell,
            pbc=[True, True, True] # Periodic embedding
        )

        return Structure.from_ase(new_atoms)
