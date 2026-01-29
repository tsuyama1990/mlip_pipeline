"""Tests for structure domain models."""

import numpy as np
import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.domain_models.structure import Structure


def test_structure_valid() -> None:
    """Test creating a valid structure."""
    s = Structure(
        formatted_formula="H2",
        numbers=[1, 1],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        cell=np.eye(3).tolist(),
        pbc=[True, True, True]
    )
    assert s.formatted_formula == "H2"
    assert len(s.positions) == 2


def test_structure_validation_error() -> None:
    """Test validation errors."""
    with pytest.raises(ValidationError):
        Structure(
            formatted_formula="H",
            numbers=[1],
            positions=[],  # Empty positions
            cell=np.eye(3).tolist(),
            pbc=[True, True, True]
        )

    with pytest.raises(ValidationError):
        Structure(
            formatted_formula="H",
            numbers=[1],
            positions=[[0.0, 0.0, 0.0]],
            cell=[[1.0, 0.0], [0.0, 1.0]],  # 2x2 cell
            pbc=[True, True, True]
        )


def test_from_ase_to_ase_roundtrip() -> None:
    """Test ASE conversion roundtrip."""
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[10, 10, 10], pbc=True)

    s = Structure.from_ase(atoms)
    assert s.formatted_formula == "H2O"
    assert s.numbers == [1, 1, 8]
    assert len(s.positions) == 3

    atoms_back = s.to_ase()
    assert atoms_back.get_chemical_formula() == "H2O"  # type: ignore[no-untyped-call]
    np.testing.assert_array_almost_equal(atoms_back.get_positions(), atoms.get_positions())  # type: ignore[no-untyped-call]
