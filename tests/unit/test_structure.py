import numpy as np
import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.structure import Structure


def test_structure_valid_creation() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    atomic_numbers = np.array([1, 1])
    cell = np.eye(3)
    pbc = np.array([True, True, True])

    struct = Structure(positions=positions, atomic_numbers=atomic_numbers, cell=cell, pbc=pbc)

    assert np.allclose(struct.positions, positions)
    assert np.array_equal(struct.atomic_numbers, atomic_numbers)
    assert np.allclose(struct.cell, cell)
    assert np.array_equal(struct.pbc, pbc)
    assert struct.energy is None
    assert struct.forces is None
    assert struct.stress is None


def test_structure_from_lists() -> None:
    struct = Structure(
        positions=[[0.0, 0.0, 0.0]],  # type: ignore[arg-type]
        atomic_numbers=[6],  # type: ignore[arg-type]
        cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],  # type: ignore[arg-type]
        pbc=[True, True, True],  # type: ignore[arg-type]
    )
    assert isinstance(struct.positions, np.ndarray)
    assert struct.positions.shape == (1, 3)


def test_structure_invalid_shapes() -> None:
    # Invalid positions shape
    with pytest.raises(ValidationError):
        Structure(
            positions=[0.0, 0.0, 0.0],  # type: ignore[arg-type] # 1D array, should be 2D
            atomic_numbers=[1],  # type: ignore[arg-type]
            cell=np.eye(3),
            pbc=[True, True, True],  # type: ignore[arg-type]
        )

    # Mismatch atoms vs positions
    with pytest.raises(ValidationError, match="Length mismatch"):
        Structure(
            positions=[[0, 0, 0]],  # type: ignore[arg-type]
            atomic_numbers=[1, 2],  # type: ignore[arg-type] # 2 atoms
            cell=np.eye(3),
            pbc=[True, True, True],  # type: ignore[arg-type]
        )

    # Invalid cell shape
    with pytest.raises(ValidationError, match="Cell must be a"):
        Structure(
            positions=[[0, 0, 0]],  # type: ignore[arg-type]
            atomic_numbers=[1],  # type: ignore[arg-type]
            cell=np.eye(2),  # 2x2
            pbc=[True, True, True],  # type: ignore[arg-type]
        )

    # Invalid PBC shape
    with pytest.raises(ValidationError, match="PBC must be a"):
        Structure(
            positions=[[0, 0, 0]],  # type: ignore[arg-type]
            atomic_numbers=[1],  # type: ignore[arg-type]
            cell=np.eye(3),
            pbc=[True, True],  # type: ignore[arg-type] # 2 elements
        )

    # Invalid atomic_numbers shape
    with pytest.raises(ValidationError, match="Atomic numbers must be a"):
        Structure(
            positions=[[0, 0, 0]],  # type: ignore[arg-type]
            atomic_numbers=[[1]],  # type: ignore[arg-type] # 2D array
            cell=np.eye(3),
            pbc=[True, True, True],  # type: ignore[arg-type]
        )


def test_structure_serialization() -> None:
    struct = Structure(
        positions=[[0.0, 0.0, 0.0]],  # type: ignore[arg-type]
        atomic_numbers=[6],  # type: ignore[arg-type]
        cell=np.eye(3),
        pbc=[True, True, True],  # type: ignore[arg-type]
        energy=-10.5,
        forces=[[0.1, 0.2, 0.3]],  # type: ignore[arg-type]
        stress=np.zeros((3, 3)),
    )

    dump = struct.model_dump()
    assert isinstance(dump["positions"], list)
    assert dump["positions"] == [[0.0, 0.0, 0.0]]
    assert dump["energy"] == -10.5
    assert dump["forces"] == [[0.1, 0.2, 0.3]]
    assert dump["stress"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert dump["pbc"] == [True, True, True]


def test_structure_forces_stress_validation() -> None:
    # Invalid forces shape
    with pytest.raises(ValidationError, match="Forces must be an"):
        Structure(
            positions=[[0, 0, 0]],  # type: ignore[arg-type]
            atomic_numbers=[1],  # type: ignore[arg-type]
            cell=np.eye(3),
            pbc=[True, True, True],  # type: ignore[arg-type]
            forces=[0.1, 0.2, 0.3],  # type: ignore[arg-type] # 1D
        )

    # Mismatch forces vs positions
    with pytest.raises(ValidationError, match="Length mismatch"):
        Structure(
            positions=[[0, 0, 0]],  # type: ignore[arg-type]
            atomic_numbers=[1],  # type: ignore[arg-type]
            cell=np.eye(3),
            pbc=[True, True, True],  # type: ignore[arg-type]
            forces=[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],  # type: ignore[arg-type] # 2 forces for 1 atom
        )

    # Stress check - we didn't strictly validate stress shape other than array, so just check it accepts array
    s = Structure(
        positions=[[0, 0, 0]],  # type: ignore[arg-type]
        atomic_numbers=[1],  # type: ignore[arg-type]
        cell=np.eye(3),
        pbc=[True, True, True],  # type: ignore[arg-type]
        stress=np.zeros(6),
    )
    assert s.stress is not None
    assert s.stress.shape == (6,)

    # Test None explicitly
    s = Structure(
        positions=[[0, 0, 0]],  # type: ignore[arg-type]
        atomic_numbers=[1],  # type: ignore[arg-type]
        cell=np.eye(3),
        pbc=[True, True, True],  # type: ignore[arg-type]
        forces=None,
        stress=None,
    )
    assert s.forces is None
    assert s.stress is None
