from ase import Atoms
from ase.neighborlist import neighbor_list


def get_neighbors(atoms: Atoms, cutoff: float) -> tuple:
    """Wrapper around ase neighbor_list to get neighbors within cutoff.

    Returns:
        i, j, d, D: Indices of first atom, indices of second atom,
                    distances, and distance vectors.
    """
    return neighbor_list("ijdD", atoms, cutoff)
