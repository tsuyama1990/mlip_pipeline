import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.training import TrainingConfig


def test_training_config_validation():
    with pytest.raises(ValidationError):
        TrainingConfig(batch_size=-1)
