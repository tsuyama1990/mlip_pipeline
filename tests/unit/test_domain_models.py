from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.potential import ExplorationResult, Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationResult


def test_structure_valid() -> None:
    s = Structure(
        symbols=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
        properties={"energy": -1.0},
        forces=np.zeros((2, 3)),
        stress=np.zeros((3, 3)),
    )
    assert len(s.symbols) == 2
    assert s.positions.shape == (2, 3)
    assert s.properties["energy"] == -1.0
    assert s.forces is not None
    assert s.forces.shape == (2, 3)
    assert s.stress is not None
    assert s.stress.shape == (3, 3)


def test_structure_invalid_positions_shape() -> None:
    with pytest.raises(ValidationError) as exc:
        Structure(
            symbols=["H"],
            positions=np.array([[0.0, 0.0]]),  # Wrong shape (1, 2)
            cell=np.eye(3),
            pbc=np.array([True, True, True]),
        )
    assert "Positions shape" in str(exc.value)


def test_structure_invalid_cell_shape() -> None:
    with pytest.raises(ValidationError) as exc:
        Structure(
            symbols=["H"],
            positions=np.array([[0.0, 0.0, 0.0]]),
            cell=np.eye(2),  # Wrong shape (2, 2)
            pbc=np.array([True, True, True]),
        )
    assert "Cell shape" in str(exc.value)


def test_structure_invalid_pbc_shape() -> None:
    with pytest.raises(ValidationError) as exc:
        Structure(
            symbols=["H"],
            positions=np.array([[0.0, 0.0, 0.0]]),
            cell=np.eye(3),
            pbc=np.array([True, True]),  # Wrong shape
        )
    assert "PBC shape" in str(exc.value)


def test_structure_invalid_forces_shape() -> None:
    with pytest.raises(ValidationError) as exc:
        Structure(
            symbols=["H"],
            positions=np.array([[0.0, 0.0, 0.0]]),
            cell=np.eye(3),
            pbc=np.array([True, True, True]),
            forces=np.zeros((2, 3)),  # Wrong shape
        )
    assert "Forces shape" in str(exc.value)


def test_structure_invalid_stress_shape() -> None:
    with pytest.raises(ValidationError) as exc:
        Structure(
            symbols=["H"],
            positions=np.array([[0.0, 0.0, 0.0]]),
            cell=np.eye(3),
            pbc=np.array([True, True, True]),
            stress=np.zeros((2, 2)),  # Wrong shape
        )
    assert "Stress shape" in str(exc.value)


def test_structure_serialization() -> None:
    s = Structure(
        symbols=["H"],
        positions=np.array([[0.0, 0.0, 0.0]]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
    )
    # Pydantic dump
    data = s.model_dump(mode="json")
    assert isinstance(data["positions"], list)
    assert data["positions"] == [[0.0, 0.0, 0.0]]

    # Round trip
    s2 = Structure(**data)
    np.testing.assert_array_equal(s.positions, s2.positions)


def test_potential_path_validation(tmp_path: Path) -> None:
    p = tmp_path / "model.yace"
    p.touch()

    pot = Potential(path=p, version="1.0")
    assert pot.path == p

    with pytest.raises(ValidationError) as exc:
        Potential(path=tmp_path / "nonexistent.yace", version="1.0")
    assert "does not exist" in str(exc.value)

    d = tmp_path / "dir"
    d.mkdir()
    with pytest.raises(ValidationError) as exc:
        Potential(path=d, version="1.0")
    assert "is not a file" in str(exc.value)


def test_exploration_result_validation() -> None:
    # Valid halted
    s = Structure(symbols=["H"], positions=np.zeros((1, 3)), cell=np.eye(3), pbc=np.ones(3))
    ExplorationResult(status="halted", structure=s)

    # Invalid halted (missing structure)
    with pytest.raises(ValidationError) as exc:
        ExplorationResult(status="halted", structure=None)
    assert "Structure must be provided" in str(exc.value)

    # Valid converged
    ExplorationResult(status="converged", structure=None)


def test_validation_result() -> None:
    vr = ValidationResult(passed=True, metrics={"rmse": 0.1})
    assert vr.passed
    assert vr.metrics["rmse"] == 0.1
