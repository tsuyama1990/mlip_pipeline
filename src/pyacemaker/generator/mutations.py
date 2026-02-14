"""Atomic mutation functions for structure generation."""

from typing import cast

import numpy as np
from ase import Atoms


def apply_strain(atoms: Atoms, strain_range: float) -> Atoms:
    """Apply random strain to the simulation box and atomic positions.

    Args:
        atoms: Input structure.
        strain_range: Maximum strain magnitude (e.g., 0.1 for 10%).

    Returns:
        Strained structure (copy).
    """
    # Create a copy to avoid modifying the original
    new_atoms = atoms.copy()  # type: ignore[no-untyped-call]

    # Generate random strain tensor components in [-strain_range, strain_range]
    strain_tensor = (np.random.random((3, 3)) - 0.5) * 2 * strain_range

    # Add identity matrix to get deformation gradient F = I + epsilon
    deformation_gradient = np.eye(3) + strain_tensor

    # Apply deformation to cell vectors
    # ASE uses row vectors for cell, so we multiply on the right
    # cell (3x3) where rows are lattice vectors.
    original_cell = new_atoms.get_cell()
    new_cell = np.dot(original_cell, deformation_gradient)

    # Set new cell and scale positions
    new_atoms.set_cell(new_cell, scale_atoms=True)

    return cast(Atoms, new_atoms)


def rattle_atoms(atoms: Atoms, stdev: float) -> Atoms:
    """Apply Gaussian noise to atomic positions.

    Args:
        atoms: Input structure.
        stdev: Standard deviation of displacement in Angstroms.

    Returns:
        Rattled structure (copy).
    """
    new_atoms = atoms.copy()  # type: ignore[no-untyped-call]
    new_atoms.rattle(stdev=stdev)
    return cast(Atoms, new_atoms)


def create_vacancy(atoms: Atoms, index: int | None = None) -> Atoms:
    """Remove an atom to create a vacancy.

    Args:
        atoms: Input structure.
        index: Index of the atom to remove. If None, remove a random atom.

    Returns:
        Structure with vacancy (copy).
    """
    new_atoms = atoms.copy()  # type: ignore[no-untyped-call]
    if len(new_atoms) == 0:
        return cast(Atoms, new_atoms)

    if index is None:
        index = int(np.random.randint(len(new_atoms)))

    if index < 0 or index >= len(new_atoms):
        # Should we raise or just ignore/clip?
        # Raise is safer.
        msg = f"Index {index} out of bounds for atoms with length {len(new_atoms)}"
        raise IndexError(msg)

    del new_atoms[index]
    return cast(Atoms, new_atoms)
