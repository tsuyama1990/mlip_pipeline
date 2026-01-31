import numpy as np
from ase.neighborlist import NeighborList

from mlip_autopipec.domain_models.structure import Structure


class CandidateManager:
    """
    Manages the processing of candidate structures:
    extraction from large MD frames and embedding into periodic boxes.
    """

    def extract_cluster(self, supercell: Structure, center_atom_index: int, radius: float) -> Structure:
        """
        Extracts a spherical cluster of atoms around a center atom.
        Uses ASE NeighborList for correct PBC handling.
        """
        atoms = supercell.to_ase()

        # cutoffs: list of radius/2 for each atom to find neighbors?
        # NeighborList takes cutoffs. If we want all neighbors within radius of center atom:
        # We can just iterate over neighbors of center atom.

        # cutoffs for NeighborList are usually atomic radii.
        # But here we want a spherical cut.
        # We can use natural_cutoffs or just a large cutoff.
        # Actually, to find neighbors within radius R, we need neighbor list with cutoff R/2 + R/2 = R.
        # So if we set all cutoffs to radius/2, then any pair with dist < R will be found.

        cutoffs = [radius / 2.0] * len(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True) # type: ignore[no-untyped-call]
        nl.update(atoms) # type: ignore[no-untyped-call]

        indices, offsets = nl.get_neighbors(center_atom_index) # type: ignore[no-untyped-call]

        # Unwrapping: get_neighbors returns offsets (pbc shifts).
        # We can reconstruct positions relative to center atom.
        # pos_neighbor = pos_original + offset @ cell
        # vector = pos_neighbor - pos_center
        # new_pos = vector (centered at 0)

        center_pos = atoms.positions[center_atom_index]
        cell = atoms.get_cell()

        new_positions = []
        new_symbols = []

        # Center atom at 0
        new_positions.append([0.0, 0.0, 0.0])
        new_symbols.append(atoms.symbols[center_atom_index])

        for i, idx in enumerate(indices):
            offset_vec = np.dot(offsets[i], cell)
            diff = atoms.positions[idx] + offset_vec - center_pos
            new_positions.append(diff)
            new_symbols.append(atoms.symbols[idx])

        # Create new Structure
        # We set a large dummy cell to avoid 0-volume issues, but no PBC.
        dummy_cell = np.eye(3) * (2 * radius + 10.0)

        cluster = Structure(
            symbols=new_symbols,
            positions=np.array(new_positions),
            cell=dummy_cell,
            pbc=(False, False, False)
        )

        # Recenter (optional, but we already centered at 0)
        # cluster.to_ase().center()

        return cluster

    def embed_cluster(self, cluster: Structure, vacuum: float = 10.0) -> Structure:
        """
        Embeds a finite cluster into a large periodic box with vacuum padding.
        """
        atoms = cluster.to_ase()
        atoms.center(vacuum=vacuum) # type: ignore[no-untyped-call]
        atoms.set_pbc((True, True, True)) # type: ignore[no-untyped-call]

        return Structure.from_ase(atoms)
