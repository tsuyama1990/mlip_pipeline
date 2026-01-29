import ase
import numpy as np
import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.structure import Candidate, CandidateStatus, Structure


def test_structure_creation_from_ase(sample_ase_atoms: ase.Atoms) -> None:
    structure = Structure.from_ase(sample_ase_atoms)

    assert structure.symbols == ["H", "H"]
    assert np.allclose(structure.positions, np.array([[0, 0, 0], [0, 0, 0.74]]))
    assert np.allclose(structure.cell, np.diag([10, 10, 10]))
    assert structure.pbc == (True, True, True)
    assert structure.properties["energy"] == -1.5


def test_structure_to_ase(sample_ase_atoms: ase.Atoms) -> None:
    structure = Structure.from_ase(sample_ase_atoms)
    atoms = structure.to_ase()

    # Ignore untyped calls from ase
    assert atoms.get_chemical_formula() == "H2" # type: ignore[no-untyped-call]
    assert np.allclose(atoms.get_positions(), sample_ase_atoms.get_positions()) # type: ignore[no-untyped-call]


def test_structure_validation_positions_shape() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Structure(
            symbols=["H"],
            positions=np.array([0, 0, 0]),  # Wrong shape (3,) instead of (1, 3)
            cell=np.eye(3),
            pbc=(True, True, True)
        )
    assert "Positions must have shape (N, 3)" in str(excinfo.value)


def test_structure_consistency_check() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Structure(
            symbols=["H", "H"],
            positions=np.array([[0, 0, 0]]),  # Only 1 position for 2 symbols
            cell=np.eye(3),
            pbc=(True, True, True)
        )
    assert "Number of symbols (2) does not match number of positions (1)" in str(excinfo.value)


def test_candidate_creation() -> None:
    candidate = Candidate(
        symbols=["H"],
        positions=np.array([[0, 0, 0]]),
        cell=np.eye(3),
        pbc=(True, True, True),
        source="test",
        priority=10
    )
    assert candidate.status == CandidateStatus.PENDING
    assert candidate.source == "test"
    assert candidate.priority == 10
