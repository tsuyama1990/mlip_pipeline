import pytest

from mlip_autopipec.components.dynamics.mock import MockDynamics
from mlip_autopipec.components.generator.mock import MockGenerator
from mlip_autopipec.components.oracle.mock import MockOracle
from mlip_autopipec.components.trainer.mock import MockTrainer
from mlip_autopipec.components.validator.mock import MockValidator
from mlip_autopipec.factory import ComponentFactory


def test_factory_creation() -> None:
    config = {"type": "mock"}

    assert isinstance(ComponentFactory.get_generator(config), MockGenerator)
    assert isinstance(ComponentFactory.get_oracle(config), MockOracle)
    assert isinstance(ComponentFactory.get_trainer(config), MockTrainer)
    assert isinstance(ComponentFactory.get_dynamics(config), MockDynamics)
    assert isinstance(ComponentFactory.get_validator(config), MockValidator)


def test_factory_invalid_role() -> None:
    with pytest.raises(ValueError, match="Unknown component role"):
        ComponentFactory.create("unknown_role", {"type": "mock"})


def test_factory_invalid_type() -> None:
    with pytest.raises(ValueError, match="Unknown component type"):
        ComponentFactory.get_generator({"type": "unknown_type"})
