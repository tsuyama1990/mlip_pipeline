import pytest
from pydantic import ValidationError

from mlip_autopipec.config.config_model import (
    DFTConfig,
    ExplorationConfig,
    GlobalConfig,
    TrainingConfig,
)


def test_default_config() -> None:
    config = GlobalConfig()
    assert config.execution_mode == "mock"
    assert config.max_cycles == 5
    assert config.exploration.strategy_name == "random"


def test_valid_config_overrides() -> None:
    config = GlobalConfig(
        execution_mode="production",
        max_cycles=10,
        exploration=ExplorationConfig(strategy_name="md", max_structures=20),
        dft=DFTConfig(encut=600.0),
        training=TrainingConfig(max_epochs=200),
    )
    assert config.execution_mode == "production"
    assert config.max_cycles == 10
    assert config.exploration.strategy_name == "md"
    assert config.dft.encut == 600.0
    assert config.training.max_epochs == 200


def test_invalid_execution_mode() -> None:
    with pytest.raises(ValidationError):
        GlobalConfig(execution_mode="invalid_mode")


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        GlobalConfig(extra_field="should_fail")  # type: ignore[call-arg]


def test_nested_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        GlobalConfig(
            exploration=ExplorationConfig(extra="fail")  # type: ignore[call-arg]
        )
