import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList


def extract_periodic_box(large_atoms: Atoms, center_index: int, cutoff: float) -> Atoms:
    """
    Cuts a cluster around 'center_index' and wraps it into a
    minimal periodic supercell that respects the cutoff radius.
    """
    # 1. Determine Box Size
    # Test requires >= 2 * cutoff.
    box_size = 2.0 * cutoff
    half_box = box_size / 2.0

    # 2. Create the target cell (Orthorhombic)
    cell = np.diag([box_size, box_size, box_size])

    # 3. Find atoms in the box
    # Search radius to cover the corners of the cube
    search_radius = box_size * np.sqrt(3) / 2.0 + 0.5

    cutoffs = [search_radius] * len(large_atoms)
    nl = NeighborList(  # type: ignore[no-untyped-call]
        cutoffs, skin=0.0, self_interaction=True, bothways=True
    )
    nl.update(large_atoms)  # type: ignore[no-untyped-call]
    indices, offsets = nl.get_neighbors(center_index)  # type: ignore[no-untyped-call]

    uc_cell = large_atoms.get_cell()  # type: ignore[no-untyped-call]
    uc_pos = large_atoms.get_positions()  # type: ignore[no-untyped-call]
    # Shift center slightly to avoid atoms falling exactly on boundary
    # This ensures we pick exactly one image of boundary atoms
    shift = np.array([1e-5, 1e-5, 1e-5])
    center_pos = uc_pos[center_index] + shift

    cluster_positions = []
    cluster_numbers = []

    for idx, offset in zip(indices, offsets, strict=False):
        # Calculate true position of the image
        real_pos = uc_pos[idx] + offset @ uc_cell
        rel_pos = real_pos - center_pos

        # Check if inside box [-L/2, L/2)
        # Using strict inequality for upper bound to avoid double counting if matches boundary
        if np.all(rel_pos >= -half_box) and np.all(rel_pos < half_box):
            cluster_positions.append(rel_pos)
            cluster_numbers.append(large_atoms.numbers[idx])

    # 4. Create new atoms
    # Shift positions to [0, L]
    final_positions = np.array(cluster_positions) + half_box

    return Atoms(numbers=cluster_numbers, positions=final_positions, cell=cell, pbc=True)
