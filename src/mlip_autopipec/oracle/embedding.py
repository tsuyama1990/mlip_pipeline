import numpy as np
from ase import Atoms


class Embedding:
    """
    Handles extraction of local clusters from larger structures and
    embedding them into periodic simulation boxes for DFT.
    """

    def __init__(self, cutoff: float, buffer: float) -> None:
        """
        Args:
            cutoff: Radius to select atoms around the center.
            buffer: Buffer size to add around the cutoff for the new box.
        """
        self.cutoff = cutoff
        self.buffer = buffer

    def cut_cluster(self, atoms: Atoms, center_index: int) -> Atoms:
        """
        Cuts a cluster around the atom at center_index and embeds it into a periodic box.

        Args:
            atoms: Source structure (ASE Atoms).
            center_index: Index of the central atom.

        Returns:
            New ASE Atoms object with the cluster centered in a new periodic box.
        """
        if center_index < 0 or center_index >= len(atoms):
            msg = f"Atom index {center_index} out of range (0-{len(atoms) - 1})"
            raise IndexError(msg)

        # Calculate distances and vectors from center atom to all others
        # using Minimum Image Convention if PBC is enabled on source.
        # Note: ASE behavior for get_distances with vector=True returns ONLY vectors in some versions/contexts
        # or (distances, vectors). To be safe, we'll check the return type or just get vectors and compute distances.
        vectors = atoms.get_distances(  # type: ignore[no-untyped-call]
            center_index,
            range(len(atoms)),
            mic=atoms.pbc.any(),
            vector=True,
        )

        # If it returned a tuple (dists, vectors), unpack it.
        # If it returned an array (vectors), use it.
        if isinstance(vectors, tuple):
            _, vectors = vectors

        # Compute distances from vectors
        dists = np.linalg.norm(vectors, axis=1)

        # Select atoms within cutoff
        mask = dists <= self.cutoff
        indices = np.where(mask)[0]

        # Define new box size
        # L = 2 * (cutoff + buffer) ensures no self-interaction for the central region
        box_length = 2.0 * (self.cutoff + self.buffer)
        cell_dims = [box_length, box_length, box_length]

        # Extract relative vectors for selected atoms
        selected_vectors = vectors[indices]
        selected_numbers = atoms.get_atomic_numbers()[indices]  # type: ignore[no-untyped-call]

        # Center the cluster in the new box
        box_center = np.array([box_length / 2.0, box_length / 2.0, box_length / 2.0])
        new_positions = box_center + selected_vectors

        # Create the new periodic structure
        return Atoms(
            numbers=selected_numbers,
            positions=new_positions,
            cell=cell_dims,
            pbc=True,
        )
