from mlip_autopipec.config.schemas.training import TrainingData
import pytest
import math

def test_training_data_validation() -> None:
    # Valid
    TrainingData(energy=-10.5, forces=[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])

    # Invalid shape
    with pytest.raises(ValueError, match="3 components"):
        TrainingData(energy=0.0, forces=[[0.0, 0.0]])

    # Invalid values (NaN)
    with pytest.raises(ValueError, match="not finite"):
        TrainingData(energy=0.0, forces=[[float('nan'), 0.0, 0.0]])

    # Invalid values (Inf)
    with pytest.raises(ValueError, match="not finite"):
        TrainingData(energy=0.0, forces=[[float('inf'), 0.0, 0.0]])
