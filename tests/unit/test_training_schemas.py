from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.training import TrainConfig, TrainingResult, TrainingData


def test_train_config_valid():
    """Test creating a valid TrainConfig."""
    config = TrainConfig(
        cutoff=5.0,
        loss_weights={"energy": 1.0, "forces": 100.0, "stress": 10.0},
        test_fraction=0.2,
        max_generations=5,
        enable_delta_learning=False,
        batch_size=50,
        ace_basis_size="small",
    )
    assert config.cutoff == 5.0
    assert config.loss_weights["forces"] == 100.0
    assert config.test_fraction == 0.2
    assert config.max_generations == 5
    assert not config.enable_delta_learning
    assert config.batch_size == 50
    assert config.ace_basis_size == "small"


def test_train_config_defaults():
    """Test TrainConfig default values."""
    config = TrainConfig()
    assert config.cutoff == 6.0
    assert config.loss_weights == {"energy": 1.0, "forces": 100.0, "stress": 10.0}
    assert config.test_fraction == 0.1
    assert config.max_generations == 10
    assert config.enable_delta_learning is True
    assert config.batch_size == 100
    assert config.ace_basis_size == "medium"


def test_train_config_invalid():
    """Test invalid TrainConfig values."""
    with pytest.raises(ValidationError):
        TrainConfig(cutoff=-1.0)

    with pytest.raises(ValidationError):
        TrainConfig(test_fraction=1.5)

    with pytest.raises(ValidationError):
        TrainConfig(batch_size=0)

    with pytest.raises(ValidationError):
        TrainConfig(ace_basis_size="huge")


def test_training_result_valid(tmp_path: Path):
    """Test creating a valid TrainingResult."""
    result = TrainingResult(
        potential_path=tmp_path / "output.yace",
        rmse_energy=0.005,
        rmse_forces=0.1,
        training_time=120.5,
        generation=1,
    )
    assert result.potential_path == tmp_path / "output.yace"
    assert result.rmse_energy == 0.005
    assert result.rmse_forces == 0.1
    assert result.training_time == 120.5
    assert result.generation == 1


def test_training_result_invalid(tmp_path: Path):
    """Test invalid TrainingResult values."""
    with pytest.raises(ValidationError):
        TrainingResult(
            potential_path=tmp_path / "output.yace",
            rmse_energy=-0.1,
            rmse_forces=0.1,
            training_time=10,
            generation=1,
        )


def test_training_data_valid():
    """Test valid TrainingData."""
    data = TrainingData(energy=-100.5, forces=[[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])
    assert data.energy == -100.5
    assert len(data.forces) == 2


def test_training_data_invalid_forces():
    """Test invalid forces shape."""
    with pytest.raises(ValidationError):
        TrainingData(
            energy=-100.5,
            forces=[[0.0, 0.0], [0.0, 0.0, -0.1]],  # 2D vector
        )


def test_training_data_type_validation():
    """Test type validation."""
    with pytest.raises(ValidationError):
        TrainingData(energy="not a float", forces=[[0.0, 0.0, 0.1]])
