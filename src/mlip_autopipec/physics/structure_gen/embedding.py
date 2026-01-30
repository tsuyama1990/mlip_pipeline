import numpy as np
import ase
from scipy.spatial import cKDTree  # type: ignore
from typing import List

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

        cell = atoms.get_cell() # type: ignore[no-untyped-call]
        pbc = atoms.get_pbc() # type: ignore[no-untyped-call]
        positions = atoms.get_positions() # type: ignore[no-untyped-call]
        symbols = atoms.get_chemical_symbols() # type: ignore[no-untyped-call]

        is_orthogonal = np.allclose(cell, np.diag(np.diag(cell)))

        # Determine indices within radius
        indices: List[int] = []

        if is_orthogonal and np.all(pbc):
            # Fast path for orthogonal periodic cells
            boxsize = np.diag(cell)
            tree = cKDTree(positions, boxsize=boxsize)
            center_pos = positions[center_index]
            indices = tree.query_ball_point(center_pos, r=self.radius)
        else:
            # General path for non-orthogonal or mixed PBC
            # Use ASE's neighbor list logic or simple distance check with mic=True
            # get_distances can handle MIC for general cells
            # We want all atoms where dist <= radius
            # This is O(N) which is fine for "large" structures ~1000 atoms,
            # but for huge ones we might want neighbor list.
            # Assuming 'Structure' is reasonable size for DFT pipeline (e.g. < 10000 atoms).

            # get_distances returns array of distances from center_index to all others
            dists = atoms.get_distances(center_index, indices=range(len(atoms)), mic=True) # type: ignore[no-untyped-call]
            indices = [i for i, d in enumerate(dists) if d <= self.radius]

        # Extract unwrapped positions relative to center
        cluster_positions_list: List[np.ndarray] = []
        cluster_symbols: List[str] = []

        center_pos = positions[center_index]

        if is_orthogonal and np.all(pbc):
             # Fast manual MIC for orthogonal
             boxsize = np.diag(cell)
             for idx in indices:
                pos = positions[idx]
                vec = pos - center_pos
                vec = vec - boxsize * np.round(vec / boxsize)
                cluster_positions_list.append(vec)
                cluster_symbols.append(symbols[idx])
        else:
            # Use ASE for general MIC vector calculation
            # get_distance with vector=True returns the vector pointing from A to B with MIC
            for idx in indices:
                # vector from center to idx
                # Note: get_distance(a, b, vector=True) returns vector from a to b.
                vec = atoms.get_distance(center_index, idx, mic=True, vector=True) # type: ignore[no-untyped-call]
                cluster_positions_list.append(vec)
                cluster_symbols.append(symbols[idx])

        cluster_positions_arr = np.array(cluster_positions_list)

        # New box size: 2 * (radius + vacuum)
        box_side = 2.0 * (self.radius + self.vacuum)
        new_cell = np.eye(3) * box_side

        # Shift positions so center is at center of new box
        center_of_box = np.array([box_side/2, box_side/2, box_side/2])
        new_positions = cluster_positions_arr + center_of_box

        new_atoms = ase.Atoms(
            symbols=cluster_symbols,
            positions=new_positions,
            cell=new_cell,
            pbc=[True, True, True] # Periodic embedding
        )

        return Structure.from_ase(new_atoms)
