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

def test_valid_structure_with_stress_and_properties() -> None:
    positions = np.array([[0.0, 0.0, 0.0]])
    cell = np.eye(3)
    species = ["H"]
    stress = np.eye(3)
    properties = {"gamma": 0.5, "origin": "test"}

    s = Structure(
        positions=positions,
        cell=cell,
        species=species,
        stress=stress,
        properties=properties
    )
    assert s.stress is not None
    assert s.stress.shape == (3, 3)
    assert s.properties["gamma"] == 0.5

def test_invalid_stress_shape() -> None:
    positions = np.array([[0.0, 0.0, 0.0]])
    cell = np.eye(3)
    species = ["H"]
    stress = np.array([0.0, 0.0]) # Invalid shape

    with pytest.raises(ValidationError) as excinfo:
        Structure(
            positions=positions,
            cell=cell,
            species=species,
            stress=stress
        )
    assert "Stress must be a (3, 3) array" in str(excinfo.value)

def test_apply_periodic_embedding_validation() -> None:
    positions = np.array([[0.0, 0.0, 0.0]])
    cell = np.eye(3) * 10.0
    species = ["H"]
    s = Structure(positions=positions, cell=cell, species=species)
    center = np.array([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="radius must be positive"):
        s.apply_periodic_embedding(center, -1.0, 0.0)

    with pytest.raises(ValueError, match="buffer must be non-negative"):
        s.apply_periodic_embedding(center, 1.0, -1.0)

    with pytest.raises(ValueError, match=r"center must be a \(3,\) array"):
        s.apply_periodic_embedding(np.array([0.0, 0.0]), 1.0, 0.0)

def test_apply_periodic_embedding_logic() -> None:
    # Create a structure: 2 atoms, one at origin, one far away
    # Use simple cubic cell 10x10x10
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    cell = np.eye(3) * 10.0
    species = ["H", "He"]
    s = Structure(positions=positions, cell=cell, species=species)

    center = np.array([0.0, 0.0, 0.0])
    radius = 2.0
    buffer = 1.0

    # We expect atoms within radius+buffer = 3.0
    # Atom at 0,0,0 is at distance 0.
    # Atom at 5,5,5 is at distance sqrt(75) ~ 8.66.

    # Note: PBC logic checks minimal image.
    # 5,5,5 is the furthest point in 10,10,10 cell from 0,0,0.
    # Image of 5,5,5:
    # 5-10 = -5. distance is still 8.66.

    new_s = s.apply_periodic_embedding(center, radius, buffer)

    # We expect 1 atom (H)
    assert len(new_s.species) == 1
    assert new_s.species[0] == "H"
