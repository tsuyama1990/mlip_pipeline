import numpy as np

from mlip_autopipec.domain_models.structure import Structure


class CandidateManager:
    """
    Manages the processing of candidate structures:
    extraction from large MD frames and embedding into periodic boxes.
    """

    def extract_cluster(self, supercell: Structure, center_atom_index: int, radius: float) -> Structure:
        """
        Extracts a spherical cluster of atoms around a center atom.
        """
        atoms = supercell.to_ase()

        # We handle PBC by using neighbor list which handles it
        # Or simpler: compute distances. But neighbor list is better for PBC.
        # cutoffs: list of radius for each atom. We only care about center.

        # Simple approach: Calculate distances from center atom to all others (considering PBC)
        # ASE has get_distances

        # However, to properly extract a cluster that 'looks' like the local environment,
        # we might want to unwrap it or keep relative positions.
        # Simplest valid approach for DFT:
        # 1. Build NeighborList
        # 2. Identify indices.
        # 3. Create new Atoms.

        # This implementation uses simple distance check which might be slow for huge systems
        # but robust enough for this cycle.
        # Actually, get_distances handles MIC (Minimum Image Convention).

        dists = atoms.get_distances(center_atom_index, range(len(atoms)), mic=True) # type: ignore[no-untyped-call]
        mask = dists <= radius
        indices = np.where(mask)[0]

        cluster_atoms = atoms[indices]

        # Center the cluster?
        # Maybe shift positions so center_atom is at origin?
        # For now, just return as is but wrapped?
        # Ideally we want a non-periodic cluster initially.
        cluster_atoms.set_pbc((False, False, False)) # type: ignore[no-untyped-call]

        # We don't set cell to zero, to avoid issues with zero-volume cells in some tools.
        # We keep the original cell (though pbc is false).
        # Or we can set it to a large dummy box to be safe.
        cluster_atoms.set_cell(np.eye(3) * (2 * radius + 10.0)) # type: ignore[no-untyped-call]
        cluster_atoms.center() # type: ignore[no-untyped-call]

        return Structure.from_ase(cluster_atoms)

    def embed_cluster(self, cluster: Structure, vacuum: float = 10.0) -> Structure:
        """
        Embeds a finite cluster into a large periodic box with vacuum padding.
        """
        atoms = cluster.to_ase()
        atoms.center(vacuum=vacuum) # type: ignore[no-untyped-call]
        atoms.set_pbc((True, True, True)) # type: ignore[no-untyped-call]

        # We might want to set a 'ghost_mask' array here if we were doing ghost atoms.
        # For now, just simple embedding.

        return Structure.from_ase(atoms)
