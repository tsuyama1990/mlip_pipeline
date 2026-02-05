import pytest
from pydantic import ValidationError

from mlip_autopipec.config.config_model import (
    DFTConfig,
    ExplorationConfig,
    GlobalConfig,
    TrainingConfig,
)


def test_default_config() -> None:
    """Test that a global config can be instantiated with defaults."""
    config = GlobalConfig()
    assert config.execution_mode == "mock"
    assert config.max_cycles == 5
    assert isinstance(config.exploration, ExplorationConfig)
    assert isinstance(config.dft, DFTConfig)
    assert isinstance(config.training, TrainingConfig)

def test_custom_config() -> None:
    """Test that custom values are respected."""
    config = GlobalConfig(
        execution_mode="real",
        max_cycles=10,
        project_name="test_project"
    )
    assert config.execution_mode == "real"
    assert config.max_cycles == 10
    assert config.project_name == "test_project"

def test_invalid_config_type() -> None:
    """Test that invalid types raise ValidationError."""
    with pytest.raises(ValidationError):
        GlobalConfig(max_cycles="five")

def test_extra_forbidden() -> None:
    """Test that extra fields are forbidden."""
    with pytest.raises(ValidationError):
        GlobalConfig(extra_field="invalid") # type: ignore
