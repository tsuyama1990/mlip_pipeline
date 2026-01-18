import pytest
from pydantic import ValidationError

from mlip_autopipec.data_models.training_data import TrainingData


def test_training_data_valid() -> None:
    data = TrainingData(
        structure_uid="uid-1",
        energy=-100.0,
        forces=[[0.0, 0.0, 0.0]],
        stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    assert data.structure_uid == "uid-1"
    assert data.energy == -100.0

def test_training_data_invalid_forces_shape() -> None:
    # Forces should be Nx3
    with pytest.raises(ValidationError) as exc:
        TrainingData(
            structure_uid="uid-2",
            forces=[[0.0, 0.0]] # 2D vector, invalid
        )
    assert "Force vector at index 0 must be size 3" in str(exc.value)

def test_training_data_forces_none_allowed() -> None:
    data = TrainingData(structure_uid="uid-3")
    assert data.forces is None
