from typing import Sequence

from ase import Atoms

from mlip_autopipec.domain_models.structure import Structure


class EmbeddingHandler:
    """
    Handles extraction of clusters and embedding in periodic boxes.
    """

    def extract_cluster(
        self, structure: Structure, indices: Sequence[int], vacuum: float = 5.0
    ) -> Structure:
        """
        Extract atoms specified by indices and wrap in a vacuum-padded periodic box.
        """
        atoms: Atoms = structure.to_ase()

        # Select atoms
        # ASE supports list of indices
        cluster = atoms[indices]

        if isinstance(cluster, Atoms):
            cluster.center(vacuum=vacuum)  # type: ignore[no-untyped-call]
            # Set PBC to True for all directions (as required by most DFT codes for "isolated" in box)
            cluster.set_pbc((True, True, True))  # type: ignore[no-untyped-call]

            return Structure.from_ase(cluster)
        else:
            # Should not happen if indices is list
            raise ValueError("Failed to extract cluster: ASE returned non-Atoms object")
