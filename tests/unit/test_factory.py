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


def test_mock_generator_iterator() -> None:
    generator = MockGenerator({})
    structures_iter = generator.generate(n_structures=5)

    # Check that it returns an iterator
    assert hasattr(structures_iter, "__iter__")
    assert hasattr(structures_iter, "__next__")

    # Check that we can iterate and get structures
    count = 0
    for s in structures_iter:
        count += 1
        assert s.positions.shape[0] == 2  # Default n_atoms
    assert count == 5
