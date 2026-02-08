import numpy as np
import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models import Structure


def test_structure_valid() -> None:
    s = Structure(
        positions=np.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        pbc=(True, True, True),
        properties={"test": 1},
    )
    assert s.positions.shape == (1, 3)


def test_structure_invalid_shapes() -> None:
    # positions don't match atoms
    with pytest.raises(ValidationError):
        Structure(
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            atomic_numbers=np.array([1]),
            cell=np.eye(3),
        )

    # cell not 3x3
    with pytest.raises(ValidationError):
        Structure(
            positions=np.array([[0.0, 0.0, 0.0]]),
            atomic_numbers=np.array([1]),
            cell=np.eye(2),
        )


def test_structure_properties_validation() -> None:
    # properties not dict
    with pytest.raises(ValidationError):
        Structure(
            positions=np.array([[0.0, 0.0, 0.0]]),
            atomic_numbers=np.array([1]),
            cell=np.eye(3),
            properties="not a dict",  # type: ignore
        )


def test_validate_labeled() -> None:
    s = Structure(
        positions=np.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
    )
    with pytest.raises(ValueError, match="missing energy"):
        s.validate_labeled()

    s.energy = -1.0
    with pytest.raises(ValueError, match="missing forces"):
        s.validate_labeled()

    s.forces = np.zeros((1, 3))
    with pytest.raises(ValueError, match="missing stress"):
        s.validate_labeled()

    s.stress = np.zeros((3, 3))
    s.validate_labeled()  # Should pass


def test_stress_voigt() -> None:
    """Test that stress in Voigt notation (6,) is allowed."""
    s = Structure(
        positions=np.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        energy=-1.0,
        forces=np.zeros((1, 3)),
        stress=np.zeros(6),
    )
    s.validate_labeled()

    # Invalid stress shape
    with pytest.raises(ValidationError):
        Structure(
            positions=np.array([[0.0, 0.0, 0.0]]),
            atomic_numbers=np.array([1]),
            cell=np.eye(3),
            stress=np.zeros(5),
        )
