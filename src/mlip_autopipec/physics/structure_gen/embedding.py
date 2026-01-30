import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from mlip_autopipec.domain_models.structure import Structure


class EmbeddingHandler:
    def embed_cluster(
        self,
        structure: Structure,
        center_index: int,
        radius: float,
        vacuum: float = 10.0,
    ) -> Structure:
        """
        Extracts a cluster around an atom and places it in a vacuum-padded box.
        Uses ASE neighbor list to handle periodic boundary conditions correctly.
        """
        atoms = structure.to_ase()

        # Get neighbors including periodic images
        # i: source index, j: neighbor index, S: shift vector, d: distance
        i_list, j_list, S_list, D_list = neighbor_list("ijSd", atoms, cutoff=radius)

        # Filter for the specific center atom
        mask = i_list == center_index
        neighbors_j = j_list[mask]
        neighbors_S = S_list[mask]

        # Initialize cluster with the center atom at (0,0,0)
        cluster_symbols = [atoms.get_chemical_symbols()[center_index]]
        cluster_positions = [np.array([0.0, 0.0, 0.0])]

        center_pos_orig = atoms.positions[center_index]
        cell = atoms.get_cell()

        for j, S in zip(neighbors_j, neighbors_S):
            # Calculate position of the neighbor image relative to the center atom
            # pos_j_image = pos_j + S @ cell
            # vector = pos_j_image - center_pos
            vec = atoms.positions[j] + S @ cell - center_pos_orig

            dist = np.linalg.norm(vec)

            # Skip the center atom itself if it appears in the neighbor list (dist ~ 0)
            if dist < 1e-4:
                continue

            cluster_symbols.append(atoms.get_chemical_symbols()[j])
            cluster_positions.append(vec)

        # Convert to numpy array
        pos_array = np.array(cluster_positions)

        # Determine the size of the cluster
        p_min = np.min(pos_array, axis=0)
        p_max = np.max(pos_array, axis=0)
        extent = p_max - p_min

        # Create a cubic box large enough to hold the cluster + vacuum
        # Using a cubic box is generally safer for DFT codes that might assume symmetries
        box_length = np.max(extent) + vacuum
        new_cell = np.eye(3) * box_length

        # Center the cluster in the new box
        bbox_center = (p_max + p_min) / 2.0
        box_center = np.array([box_length, box_length, box_length]) / 2.0
        shift = box_center - bbox_center

        pos_array += shift

        # Create new Atoms object
        cluster_atoms = Atoms(
            symbols=cluster_symbols, positions=pos_array, cell=new_cell, pbc=True
        )

        return Structure.from_ase(cluster_atoms)
