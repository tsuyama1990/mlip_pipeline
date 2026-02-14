"""Tests for Trainer configuration."""

import pytest
from pydantic import ValidationError

from pyacemaker.core.config import TrainerConfig


def test_trainer_config_defaults() -> None:
    """Test default values for TrainerConfig."""
    config = TrainerConfig()
    assert config.cutoff == 5.0
    assert config.order == 3
    assert config.basis_size == (15, 5)
    assert config.delta_learning == "zbl"
    assert config.max_epochs == 500
    assert config.batch_size == 100


def test_trainer_config_valid_delta_learning() -> None:
    """Test valid delta learning options."""
    for mode in ["zbl", "lj", "none", "ZBL", "LJ", "NONE"]:
        config = TrainerConfig(delta_learning=mode)
        assert config.delta_learning == mode.lower()


def test_trainer_config_invalid_delta_learning() -> None:
    """Test invalid delta learning option."""
    with pytest.raises(ValidationError) as excinfo:
        TrainerConfig(delta_learning="invalid")
    assert "Invalid delta_learning" in str(excinfo.value)
