import numpy as np
from ase import Atoms

from mlip_autopipec.domain_models.structure import Structure


class EmbeddingHandler:
    """Handles extraction of clusters from larger structures for DFT embedding."""

    def extract_cluster(
        self,
        structure: Structure,
        center_index: int,
        radius: float,
        vacuum: float = 5.0,
    ) -> Structure:
        """
        Extracts a cluster of atoms around a center atom and places it in a vacuum-padded box.

        Args:
            structure: The source structure (e.g., from MD).
            center_index: Index of the central atom.
            radius: Cutoff radius for including neighbors.
            vacuum: Amount of vacuum padding (in Angstroms) on each side.

        Returns:
            A new Structure containing the cluster in a large periodic box.
        """
        atoms = structure.to_ase()

        # Get distances from center atom to all others, respecting PBC
        # We can use atoms.get_distances which handles MIC if pbc is set
        # But get_distances(i, indices, mic=True)

        # indices of all atoms
        all_indices = list(range(len(atoms)))

        # distances from center_index to all atoms
        distances = atoms.get_distances( # type: ignore[no-untyped-call]
            center_index,
            all_indices,
            mic=True
        )

        # Select atoms within radius
        mask = distances <= radius
        cluster_indices = np.where(mask)[0]

        # Create cluster atoms
        cluster_atoms = atoms[cluster_indices]

        # Center the cluster
        # We need to ensure the atoms are contiguous (not split across PBC) before centering
        # However, extracting by distance with MIC usually gives relative vectors.
        # But atoms[indices] copies positions which might be wrapped.
        # Better approach:
        # 1. Get vectors from center to neighbors using get_distances(vector=True)
        # 2. Reconstruct positions relative to center at (0,0,0)

        vectors = atoms.get_distances( # type: ignore[no-untyped-call]
            center_index,
            cluster_indices,
            mic=True,
            vector=True
        )

        # vectors is (N_cluster, 3) pointing FROM center TO neighbor
        # So neighbor_pos = center_pos + vector
        # But we want to place center at origin (0,0,0) first, then shift to middle of new box.

        new_positions = vectors # vectors are relative to center

        # Define new box size
        # Span of the cluster
        min_pos = np.min(new_positions, axis=0)
        max_pos = np.max(new_positions, axis=0)
        span = max_pos - min_pos

        # New cell size: span + 2 * vacuum
        # Ensure cubic or orthorhombic
        cell_lengths = span + 2 * vacuum
        # Make it safe (at least some minimum size)
        cell_lengths = np.maximum(cell_lengths, [vacuum * 2] * 3)

        new_cell = np.diag(cell_lengths)

        # Shift positions to center of new box
        # Center of box is cell_lengths / 2
        # Center of cluster (which is the center atom, at 0,0,0) should be at box center?
        # Or geometric center of cluster?
        # Usually putting the central atom at the center of the box is safest.

        box_center = cell_lengths / 2.0
        shifted_positions = new_positions + box_center

        new_atoms = Atoms(
            symbols=cluster_atoms.get_chemical_symbols(), # type: ignore[no-untyped-call]
            positions=shifted_positions,
            cell=new_cell,
            pbc=[True, True, True] # Periodic embedding
        )

        return Structure.from_ase(new_atoms)
