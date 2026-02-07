import numpy as np
import pytest

from mlip_autopipec.domain_models import Structure


def test_structure_valid() -> None:
    s = Structure(
        symbols=["H", "H"],
        positions=np.zeros((2, 3)),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
    )
    assert s.positions.shape == (2, 3)


def test_structure_list_input() -> None:
    s = Structure(
        symbols=["H"],
        positions=[[0.0, 0.0, 0.0]],  # type: ignore[arg-type]
        cell=np.eye(3).tolist(),  # type: ignore[arg-type]
        pbc=[True, True, True],  # type: ignore[arg-type]
    )
    assert isinstance(s.positions, np.ndarray)
    assert s.positions.shape == (1, 3)
    assert s.cell.shape == (3, 3)


def test_structure_invalid_shape() -> None:
    with pytest.raises(ValueError, match="Positions must be"):
        Structure(
            symbols=["H", "H"],
            positions=np.zeros((2, 2)),  # Wrong shape
            cell=np.eye(3),
            pbc=np.array([True, True, True]),
        )


def test_structure_invalid_cell_shape() -> None:
    with pytest.raises(ValueError, match="Cell must be"):
        Structure(
            symbols=["H"],
            positions=np.zeros((1, 3)),
            cell=np.zeros((2, 2)),  # Wrong shape
            pbc=np.array([True, True, True]),
        )


def test_structure_mismatch() -> None:
    with pytest.raises(ValueError, match="Number of positions"):
        Structure(
            symbols=["H"],
            positions=np.zeros((2, 3)),  # Mismatch N=1 vs 2
            cell=np.eye(3),
            pbc=np.array([True, True, True]),
        )


def test_structure_serialization() -> None:
    s = Structure(
        symbols=["H"],
        positions=[[0.0, 0.0, 0.0]],  # type: ignore[arg-type]
        cell=np.eye(3),
        pbc=[True, True, True],  # type: ignore[arg-type]
    )
    json_str = s.model_dump_json()
    assert "positions" in json_str
    # Pydantic v2 might format spaces differently, but key should be there.
    assert "symbols" in json_str
