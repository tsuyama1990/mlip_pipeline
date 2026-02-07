import numpy as np


def get_reciprocal_lattice(cell: np.ndarray) -> np.ndarray:
    """
    Calculate reciprocal lattice vectors.
    B = 2 * pi * (A^-1)^T
    """
    # cell rows are lattice vectors a1, a2, a3
    # inv(cell) columns are reciprocal basis (without 2pi)
    # inv(cell).T rows are reciprocal basis
    return 2 * np.pi * np.linalg.inv(cell).T


def kspacing_to_grid(cell: np.ndarray, kspacing: float) -> tuple[int, int, int]:
    """
    Convert k-spacing (1/Angstrom) to k-point grid (Nx, Ny, Nz).
    N_i = max(1, ceil(|b_i| / kspacing))
    """
    if kspacing <= 0:
        msg = "kspacing must be positive"
        raise ValueError(msg)

    recip_cell = get_reciprocal_lattice(cell)
    recip_lengths = np.linalg.norm(recip_cell, axis=1)

    grid = np.ceil(recip_lengths / kspacing).astype(int)
    # Ensure at least 1 k-point
    grid = np.maximum(grid, 1)

    return (int(grid[0]), int(grid[1]), int(grid[2]))
