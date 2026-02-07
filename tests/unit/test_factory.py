import pytest

from mlip_autopipec.domain_models import (
    MockDynamicsConfig,
    MockGeneratorConfig,
    MockOracleConfig,
    MockSelectorConfig,
    MockTrainerConfig,
    MockValidatorConfig,
)
from mlip_autopipec.factory import (
    create_dynamics,
    create_generator,
    create_oracle,
    create_selector,
    create_trainer,
    create_validator,
)


def test_create_mock_oracle() -> None:
    config = MockOracleConfig(type="mock", noise_level=0.1)
    oracle = create_oracle(config)
    assert oracle is not None


def test_create_mock_trainer() -> None:
    config = MockTrainerConfig(type="mock")
    trainer = create_trainer(config)
    assert trainer is not None


def test_create_mock_dynamics() -> None:
    config = MockDynamicsConfig(type="mock")
    dynamics = create_dynamics(config)
    assert dynamics is not None


def test_create_mock_generator() -> None:
    config = MockGeneratorConfig(type="mock")
    generator = create_generator(config)
    assert generator is not None


def test_create_mock_validator() -> None:
    config = MockValidatorConfig(type="mock")
    validator = create_validator(config)
    assert validator is not None


def test_create_mock_selector() -> None:
    config = MockSelectorConfig(type="mock")
    selector = create_selector(config)
    assert selector is not None

def test_create_unknown_oracle() -> None:
    from dataclasses import dataclass
    @dataclass
    class DummyConfig:
        type: str = "unknown"

    from mlip_autopipec.factory import create_oracle
    with pytest.raises(ValueError, match="Unknown oracle type"):
        create_oracle(DummyConfig())  # type: ignore
