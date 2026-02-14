"""Tests for atomic mutation functions."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from pyacemaker.generator.mutations import apply_strain, create_vacancy, rattle_atoms


def test_apply_strain() -> None:
    """Test that strain changes cell volume and positions."""
    # Use cubic=True to get 4 atoms, ensuring some are not at origin
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    original_volume = atoms.get_volume()  # type: ignore[no-untyped-call]
    original_positions = atoms.get_positions().copy()  # type: ignore[no-untyped-call]

    # Apply large strain to guarantee volume change
    strained = apply_strain(atoms, strain_range=0.2)

    # Check volume changed
    assert not np.isclose(strained.get_volume(), original_volume)  # type: ignore[no-untyped-call]

    # Check positions changed (relative to cell basis they might be same if fractional,
    # but cartesian should change unless strain is identity which is unlikely)
    assert not np.allclose(strained.get_positions(), original_positions)  # type: ignore[no-untyped-call]

    # Check atoms object is a copy
    assert strained is not atoms


def test_rattle_atoms() -> None:
    """Test that rattle changes positions but not cell."""
    atoms = bulk("Cu", "fcc", a=3.6)
    original_volume = atoms.get_volume()  # type: ignore[no-untyped-call]
    original_positions = atoms.get_positions().copy()  # type: ignore[no-untyped-call]

    rattled = rattle_atoms(atoms, stdev=0.1)

    # Volume should be conserved
    assert np.isclose(rattled.get_volume(), original_volume)  # type: ignore[no-untyped-call]

    # Positions should be different
    assert not np.allclose(rattled.get_positions(), original_positions)  # type: ignore[no-untyped-call]

    # Check atoms object is a copy
    assert rattled is not atoms


def test_create_vacancy() -> None:
    """Test vacancy creation removes an atom."""
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)  # 4 atoms
    original_len = len(atoms)

    vacancy_structure = create_vacancy(atoms)

    assert len(vacancy_structure) == original_len - 1

    # Check atoms object is a copy
    assert vacancy_structure is not atoms


def test_create_vacancy_edge_cases() -> None:
    """Test edge cases for vacancy creation."""
    # Empty atoms
    empty = Atoms()
    res = create_vacancy(empty)
    assert len(res) == 0

    # Invalid index
    atoms = bulk("Cu")
    with pytest.raises(IndexError):
        create_vacancy(atoms, index=100)
