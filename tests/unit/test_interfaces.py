import pytest

from mlip_autopipec.interfaces import (
    BaseDynamics,
    BaseOracle,
    BaseSelector,
    BaseStructureGenerator,
    BaseTrainer,
    BaseValidator,
)


def test_base_oracle_abstract() -> None:
    with pytest.raises(TypeError):
        BaseOracle()  # type: ignore[abstract]


def test_base_trainer_abstract() -> None:
    with pytest.raises(TypeError):
        BaseTrainer()  # type: ignore[abstract]


def test_base_dynamics_abstract() -> None:
    with pytest.raises(TypeError):
        BaseDynamics()  # type: ignore[abstract]


def test_base_generator_abstract() -> None:
    with pytest.raises(TypeError):
        BaseStructureGenerator()  # type: ignore[abstract]


def test_base_validator_abstract() -> None:
    with pytest.raises(TypeError):
        BaseValidator()  # type: ignore[abstract]


def test_base_selector_abstract() -> None:
    with pytest.raises(TypeError):
        BaseSelector()  # type: ignore[abstract]
