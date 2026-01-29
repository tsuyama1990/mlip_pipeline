"""Tests for structure domain models."""

import numpy as np
import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.domain_models.structure import Candidate, CandidateStatus, Structure


def test_structure_valid() -> None:
    """Test creating a valid structure."""
    s = Structure(
        formatted_formula="H2",
        numbers=[1, 1],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        cell=np.eye(3).tolist(),
        pbc=[True, True, True],
        energy=-13.6,
        forces=[[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]],
        stress=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        uncertainty=0.01
    )
    assert s.formatted_formula == "H2"
    assert len(s.positions) == 2
    assert s.energy == -13.6
    assert s.uncertainty == 0.01


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
    """Test ASE conversion roundtrip with properties."""
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[10, 10, 10], pbc=True)
    atoms.info['energy'] = -500.0
    atoms.arrays['forces'] = np.random.rand(3, 3)
    atoms.info['stress'] = np.zeros(6)

    s = Structure.from_ase(atoms)
    assert s.formatted_formula == "H2O"
    assert s.numbers == [1, 1, 8]
    assert len(s.positions) == 3
    assert s.energy == -500.0
    assert s.forces is not None
    assert len(s.forces) == 3

    atoms_back = s.to_ase()
    assert atoms_back.get_chemical_formula() == "H2O"  # type: ignore[no-untyped-call]
    np.testing.assert_array_almost_equal(
        atoms_back.get_positions(),  # type: ignore[no-untyped-call]
        atoms.get_positions()  # type: ignore[no-untyped-call]
    )
    assert atoms_back.info['energy'] == -500.0
    np.testing.assert_array_almost_equal(atoms_back.arrays['forces'], atoms.arrays['forces'])


def test_candidate_creation() -> None:
    """Test creating a Candidate."""
    s = Structure(
        formatted_formula="Si",
        numbers=[14],
        positions=[[0, 0, 0]],
        cell=np.eye(3).tolist(),
        pbc=[True, True, True]
    )

    c = Candidate(
        **s.model_dump(),
        source="test",
        status=CandidateStatus.PENDING
    )
    assert c.status == CandidateStatus.PENDING
    assert c.source == "test"

    # Test Enum
    c.status = CandidateStatus.TRAINING
    assert c.status == "TRAINING"
