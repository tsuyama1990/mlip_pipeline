import numpy as np
import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.structure import Structure


def test_valid_structure() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    cell = np.eye(3)
    species = ["H", "H"]

    s = Structure(positions=positions, cell=cell, species=species)
    assert s.positions.shape == (2, 3)
    assert s.cell.shape == (3, 3)
    assert len(s.species) == 2
    assert s.forces is None

def test_valid_structure_with_forces() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    cell = np.eye(3)
    species = ["H", "H"]
    forces = np.zeros((2, 3))

    s = Structure(positions=positions, cell=cell, species=species, forces=forces)
    assert s.forces is not None
    assert s.forces.shape == (2, 3)

def test_structure_explicit_none_forces() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    cell = np.eye(3)
    species = ["H", "H"]

    # Passing None explicitly should trigger validator if configured,
    # but strictly speaking we just want to ensure it works.
    s = Structure(positions=positions, cell=cell, species=species, forces=None)
    assert s.forces is None

def test_invalid_positions_shape() -> None:
    positions = np.array([0.0, 0.0, 0.0]) # 1D array, invalid
    cell = np.eye(3)
    species = ["H"]

    with pytest.raises(ValidationError):
        Structure(positions=positions, cell=cell, species=species)

    positions = np.array([[0.0, 0.0], [1.0, 1.0]]) # (2, 2), invalid
    with pytest.raises(ValidationError):
        Structure(positions=positions, cell=cell, species=species)

def test_invalid_cell_shape() -> None:
    positions = np.array([[0.0, 0.0, 0.0]])
    cell = np.eye(2) # (2, 2), invalid
    species = ["H"]

    with pytest.raises(ValidationError):
        Structure(positions=positions, cell=cell, species=species)

def test_invalid_forces_shape() -> None:
    positions = np.array([[0.0, 0.0, 0.0]])
    cell = np.eye(3)
    species = ["H"]
    forces = np.array([0.0, 0.0, 0.0]) # 1D array, invalid

    with pytest.raises(ValidationError) as excinfo:
        Structure(
            positions=positions,
            cell=cell,
            species=species,
            forces=forces
        )
    # Check that the error message contains our custom message
    assert "Forces must be an (N, 3) array" in str(excinfo.value)
