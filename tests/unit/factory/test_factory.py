import pytest

from mlip_autopipec.domain_models import (
    MockDynamicsConfig,
    MockOracleConfig,
    MockTrainerConfig,
)
from mlip_autopipec.factory import create_component
from mlip_autopipec.infrastructure.mocks import MockDynamics, MockOracle, MockTrainer


def test_create_oracle() -> None:
    config = MockOracleConfig()
    component = create_component(config)
    assert isinstance(component, MockOracle)


def test_create_trainer() -> None:
    config = MockTrainerConfig()
    component = create_component(config)
    assert isinstance(component, MockTrainer)


def test_create_dynamics() -> None:
    config = MockDynamicsConfig()
    component = create_component(config)
    assert isinstance(component, MockDynamics)


def test_create_unknown_config() -> None:
    class UnknownConfig:
        pass

    config = UnknownConfig()
    with pytest.raises(ValueError, match="Unknown config type"):
        create_component(config)
