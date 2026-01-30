import numpy as np
from ase import Atoms  # type: ignore

from mlip_autopipec.domain_models.structure import Structure


class EmbeddingHandler:
    """Handles embedding of local environments into periodic boxes."""

    @staticmethod
    def extract_cluster(
        structure: Structure, center_index: int, radius: float, vacuum: float
    ) -> Structure:
        """
        Extract a cluster of atoms around a center atom and wrap it in a box.

        Args:
            structure: Source structure.
            center_index: Index of the central atom.
            radius: Cutoff radius for the cluster.
            vacuum: Amount of vacuum padding around the cluster (total buffer).

        Returns:
            A new Structure containing the cluster centered in a vacuum-padded box.
        """
        atoms = structure.to_ase()

        # Get vectors from center to all other atoms, respecting PBC (mic=True)
        # vectors: array of shape (N, 3) where vectors[i] is the vector from center to atom i
        vectors = atoms.get_distances(  # type: ignore[no-untyped-call]
            center_index,
            list(range(len(atoms))),
            mic=True,
            vector=True,
        )

        # Calculate distances
        distances = np.linalg.norm(vectors, axis=1)

        # Filter atoms within radius
        mask = distances < radius
        indices = np.where(mask)[0]

        cluster_vectors = vectors[indices]
        cluster_symbols = [atoms.get_chemical_symbols()[i] for i in indices] # type: ignore[no-untyped-call]

        # Calculate new box size
        # The cluster spans roughly [-radius, radius].
        # Box length should be at least 2*radius + 2*vacuum to avoid self-interaction.
        # We interpret 'vacuum' as padding on each side.
        box_length = 2 * radius + 2 * vacuum
        cell = np.eye(3) * box_length

        # Shift positions so center is at the center of the box
        # center atom corresponds to vector [0,0,0] (distance 0)
        box_center = np.array([box_length / 2, box_length / 2, box_length / 2])
        new_positions = cluster_vectors + box_center

        return Structure(
            symbols=cluster_symbols,
            positions=new_positions,
            cell=cell,
            pbc=(True, True, True),
            properties={"original_indices": indices.tolist()},
        )
