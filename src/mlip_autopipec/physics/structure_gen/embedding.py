import numpy as np
from ase.neighborlist import NeighborList
from mlip_autopipec.domain_models.structure import Structure


class EmbeddingHandler:
    def extract_periodic_box(
        self,
        structure: Structure,
        center_index: int,
        box_length: float,
    ) -> Structure:
        """
        Extracts a cubic periodic box centered at a specific atom.
        Implements Periodic Embedding (Spec 3.2).

        This extracts atoms from the original structure (supercell) that fall within
        a cubic box of size `box_length` centered at `structure.positions[center_index]`.
        The resulting structure is a periodic cubic cell of size `box_length`.
        """
        atoms = structure.to_ase()
        center_pos = atoms.positions[center_index]
        cell = atoms.get_cell()

        # Calculate search radius to cover the box diagonal
        # We add a small epsilon to ensure we catch boundary atoms
        radius = (box_length * np.sqrt(3) / 2.0) + 0.1

        # NeighborList to find atoms within the bounding sphere of the box
        cutoffs = [radius / 2.0] * len(atoms)
        nl = NeighborList(cutoffs, self_interaction=True, bothways=True) # type: ignore[no-untyped-call]
        nl.update(atoms) # type: ignore[no-untyped-call]

        indices, offsets = nl.get_neighbors(center_index) # type: ignore[no-untyped-call]

        new_positions = []
        new_symbols = []
        distances = []

        # We define the new box from -L/2 to L/2 relative to center
        half_box = box_length / 2.0

        for i, idx in enumerate(indices):
            # Vector from center to neighbor (handling PBC of original structure)
            vec = atoms.positions[idx] + np.dot(offsets[i], cell) - center_pos

            # Check if inside the box
            if np.all(np.abs(vec) <= half_box):
                # Shift to new box coordinates (0 to L)
                # The new box is centered at L/2, L/2, L/2 relative to the atom's origin (0,0,0) in the new frame?
                # No, we want the atom at the center of the box usually,
                # but standard unit cells start at 0.
                # So we map [-L/2, L/2] to [0, L].
                pos_in_box = vec + half_box

                new_positions.append(pos_in_box)
                new_symbols.append(atoms.symbols[idx])
                distances.append(float(np.linalg.norm(vec)))

        # Create Structure
        new_cell = np.eye(3) * box_length

        # Fallback if somehow empty (should at least have center)
        if not new_positions:
             new_positions = [np.array([half_box, half_box, half_box])]
             new_symbols = [atoms.symbols[center_index]]
             distances = [0.0]

        struct = Structure(
            symbols=new_symbols,
            positions=np.array(new_positions),
            cell=new_cell,
            pbc=(True, True, True)
        )
        # Store distance from center for later Force Masking
        struct.arrays['cluster_dist'] = np.array(distances)

        return struct
