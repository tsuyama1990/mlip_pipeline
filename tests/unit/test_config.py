import pytest
from pydantic import ValidationError

from mlip_autopipec.config.config_model import (
    DFTConfig,
    ExplorationConfig,
    GlobalConfig,
    TrainingConfig,
)
from mlip_autopipec.config.defaults import get_default_config


def test_valid_config() -> None:
    config = GlobalConfig(
        project_name="test_project",
        execution_mode="mock",
        cycles=1,
        dft=DFTConfig(calculator="lj"),
        training=TrainingConfig(potential_type="ace"),
        exploration=ExplorationConfig(strategy="random"),
    )
    assert config.project_name == "test_project"
    assert config.dft.calculator == "lj"


def test_invalid_cycles() -> None:
    with pytest.raises(ValidationError):
        GlobalConfig(
            project_name="test",
            dft=DFTConfig(calculator="lj"),
            training=TrainingConfig(potential_type="ace"),
            exploration=ExplorationConfig(strategy="random"),
            cycles=0,
        )


def test_invalid_cutoff() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(potential_type="ace", cutoff=-1.0)


def test_default_config() -> None:
    config = get_default_config()
    assert config.cycles == 3
    assert config.execution_mode == "mock"
