import logging
import numpy as np
from ase import Atoms
from ase.geometry import get_distances

logger = logging.getLogger(__name__)


class ClusterEmbedder:
    """
    Extracts a local cluster around a defect/atom and embeds it in a periodic box for DFT.
    """

    def __init__(self, cutoff: float, vacuum: float = 10.0):
        self.cutoff = cutoff
        self.vacuum = vacuum

    def embed(
        self,
        atoms: Atoms,
        center_index: int | None = None,
        center_position: np.ndarray | None = None,
    ) -> Atoms:
        """
        Cuts a spherical cluster and places it in a vacuum box.
        """
        if center_index is None and center_position is None:
            msg = "Must provide center_index or center_position"
            raise ValueError(msg)

        # Work on a copy
        work_atoms = atoms.copy()

        # Determine center coordinates
        if center_index is not None:
            center = work_atoms.positions[center_index]
        else:
            center = np.array(center_position)

        # Calculate distances considering PBC
        # We calculate distance from 'center' to all atoms.
        # If the box is small compared to cutoff, we might need to supercell.
        # For safety, we assume input atoms is 'large enough' or we replicate.
        # A robust way is to use 'nl' or just simple mic check if atoms.cell is reasonable.

        # Using get_distances from ase.geometry
        # It takes p1 (M, 3), p2 (N, 3), cell, pbc.
        # p1 = [center], p2 = positions

        positions = work_atoms.get_positions()

        # Calculate distances with MIC
        _, dists = get_distances(
            p1=center[None, :],
            p2=positions,
            cell=work_atoms.get_cell(),
            pbc=work_atoms.get_pbc()
        )
        # dists is (1, N)
        dists = dists.flatten()

        mask = dists <= self.cutoff
        cluster_atoms = work_atoms[mask]

        # Now center the cluster in a new box
        # 1. Define box size
        box_size = 2 * self.cutoff + self.vacuum
        new_cell = np.eye(3) * box_size

        # 2. Shift positions relative to the center
        # We need the vector from center to atom, respecting MIC.
        # get_distances returns distance magnitude. We need vectors.
        # ase.geometry.get_distances returns (D, D2) where D is vector array.

        vectors, _ = get_distances(
            p1=center[None, :],
            p2=cluster_atoms.get_positions(),
            cell=work_atoms.get_cell(),
            pbc=work_atoms.get_pbc()
        )
        # vectors is (1, M, 3) -> vector from center to atom
        relative_pos = vectors[0]

        # 3. Create new atoms
        # Place center at box center
        box_center = np.array([box_size / 2] * 3)
        new_positions = box_center + relative_pos

        embedded = Atoms(
            symbols=cluster_atoms.get_chemical_symbols(),
            positions=new_positions,
            cell=new_cell,
            pbc=[True, True, True] # DFT usually prefers periodic
        )

        return embedded
