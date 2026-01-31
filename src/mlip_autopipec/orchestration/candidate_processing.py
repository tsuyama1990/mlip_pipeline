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
        Tags atoms outside the core radius (radius - buffer) as ghosts if buffer is considered.
        However, for extraction, we just take everything within radius.
        The caller is responsible for providing R_cut + R_buffer as the radius.

        We will tag atoms based on distance from center for later Force Masking.
        """
        atoms = supercell.to_ase()

        # Determine neighbors within radius (cutoff + buffer)
        cutoffs = [radius / 2.0] * len(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True) # type: ignore[no-untyped-call]
        nl.update(atoms) # type: ignore[no-untyped-call]

        indices, offsets = nl.get_neighbors(center_atom_index) # type: ignore[no-untyped-call]

        # Include center atom
        center_pos = atoms.positions[center_atom_index]
        cell = atoms.get_cell()

        new_positions = []
        new_symbols = []
        distances = []

        # Add center atom
        new_positions.append([0.0, 0.0, 0.0])
        new_symbols.append(atoms.symbols[center_atom_index])
        distances.append(0.0)

        for i, idx in enumerate(indices):
            offset_vec = np.dot(offsets[i], cell)
            diff = atoms.positions[idx] + offset_vec - center_pos
            dist = np.linalg.norm(diff)

            new_positions.append(diff)
            new_symbols.append(atoms.symbols[idx])
            distances.append(float(dist)) # Cast to float for Mypy

        # Create new Structure
        # We set a large dummy cell to avoid 0-volume issues, but no PBC.
        dummy_cell = np.eye(3) * (2 * radius + 10.0)

        cluster = Structure(
            symbols=new_symbols,
            positions=np.array(new_positions),
            cell=dummy_cell,
            pbc=(False, False, False)
        )

        # Add distances to properties for later masking
        cluster.arrays['cluster_dist'] = np.array(distances)

        return cluster

    def embed_cluster(self, cluster: Structure, vacuum: float = 10.0) -> Structure:
        """
        Embeds a finite cluster into a large periodic box with vacuum padding.
        """
        atoms = cluster.to_ase()
        atoms.center(vacuum=vacuum) # type: ignore[no-untyped-call]
        atoms.set_pbc((True, True, True)) # type: ignore[no-untyped-call]

        return Structure.from_ase(atoms)
