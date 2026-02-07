import numpy as np
import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.structure import Structure


def test_structure_valid_numpy() -> None:
    """Test valid initialization with numpy arrays."""
    s = Structure(
        positions=np.zeros((2, 3)),
        atomic_numbers=np.array([1, 1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
    )
    assert isinstance(s.positions, np.ndarray)
    assert s.positions.shape == (2, 3)
    assert s.atomic_numbers.shape == (2,)
    assert s.cell.shape == (3, 3)
    assert s.pbc.shape == (3,)


def test_structure_valid_list() -> None:
    """Test valid initialization with lists (conversion)."""
    s = Structure(
        positions=[[0, 0, 0], [1, 1, 1]],  # type: ignore
        atomic_numbers=[1, 1],  # type: ignore
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # type: ignore
        pbc=[True, True, True],  # type: ignore
    )
    assert isinstance(s.positions, np.ndarray)
    assert s.positions.shape == (2, 3)
    assert isinstance(s.atomic_numbers, np.ndarray)
    assert isinstance(s.cell, np.ndarray)
    assert isinstance(s.pbc, np.ndarray)


def test_structure_invalid_positions_shape() -> None:
    """Test validation error for invalid positions shape."""
    with pytest.raises(ValidationError) as excinfo:
        Structure(
            positions=np.zeros((2, 2)),
            atomic_numbers=np.array([1, 1]),
            cell=np.eye(3),
            pbc=np.array([True, True, True]),
        )
    assert "positions must be Nx3 array" in str(excinfo.value)


def test_structure_invalid_atomic_numbers_shape() -> None:
    """Test validation error for invalid atomic_numbers shape."""
    with pytest.raises(ValidationError) as excinfo:
        Structure(
            positions=np.zeros((2, 3)),
            atomic_numbers=np.zeros((2, 2)),
            cell=np.eye(3),
            pbc=np.array([True, True, True]),
        )
    assert "atomic_numbers must be 1D array" in str(excinfo.value)


def test_structure_invalid_cell_shape() -> None:
    """Test validation error for invalid cell shape."""
    with pytest.raises(ValidationError) as excinfo:
        Structure(
            positions=np.zeros((2, 3)),
            atomic_numbers=np.array([1, 1]),
            cell=np.zeros((2, 2)),
            pbc=np.array([True, True, True]),
        )
    assert "cell must be 3x3 array" in str(excinfo.value)


def test_structure_invalid_pbc_shape() -> None:
    """Test validation error for invalid pbc shape."""
    with pytest.raises(ValidationError) as excinfo:
        Structure(
            positions=np.zeros((2, 3)),
            atomic_numbers=np.array([1, 1]),
            cell=np.eye(3),
            pbc=np.array([True, True]),
        )
    assert "pbc must be array of 3 booleans" in str(excinfo.value)


def test_structure_invalid_type_positions() -> None:
    """Test validation error for invalid positions type."""
    with pytest.raises((ValidationError, TypeError)) as excinfo:
        Structure(
            positions="invalid",  # type: ignore
            atomic_numbers=np.array([1, 1]),
            cell=np.eye(3),
            pbc=np.array([True, True, True]),
        )
    assert "positions must be a numpy array or list" in str(excinfo.value)


def test_structure_stress_and_forces() -> None:
    """Test valid initialization with stress and forces."""
    s = Structure(
        positions=np.zeros((2, 3)),
        atomic_numbers=np.array([1, 1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
        forces=[[0, 0, 0], [0, 0, 0]],  # type: ignore
        stress=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # type: ignore
    )
    assert isinstance(s.forces, np.ndarray)
    assert isinstance(s.stress, np.ndarray)


def test_structure_serialization() -> None:
    """Test serialization of numpy arrays to lists."""
    s = Structure(
        positions=[[0, 0, 0], [1, 1, 1]],  # type: ignore
        atomic_numbers=[1, 1],  # type: ignore
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # type: ignore
        pbc=[True, True, True],  # type: ignore
        forces=[[0, 0, 0], [0, 0, 0]],  # type: ignore
        stress=np.eye(3),
    )
    d = s.model_dump(mode="json")
    assert isinstance(d["positions"], list)
    assert isinstance(d["atomic_numbers"], list)
    assert isinstance(d["cell"], list)
    assert isinstance(d["pbc"], list)
    assert isinstance(d["forces"], list)
    assert isinstance(d["stress"], list)
    assert d["positions"] == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
