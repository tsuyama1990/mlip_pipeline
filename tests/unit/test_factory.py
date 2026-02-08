import pytest

from mlip_autopipec.components.dynamics.mock import MockDynamics
from mlip_autopipec.components.generator.mock import MockGenerator
from mlip_autopipec.components.oracle.mock import MockOracle
from mlip_autopipec.components.trainer.mock import MockTrainer
from mlip_autopipec.components.validator.mock import MockValidator
from mlip_autopipec.domain_models.config import (
    ComponentConfig,
    DynamicsConfig,
    GeneratorConfig,
    OracleConfig,
    TrainerConfig,
    ValidatorConfig,
)
from mlip_autopipec.factory import ComponentFactory


def test_factory_creation() -> None:
    # Use valid configs for factory creation
    gen_config = GeneratorConfig(cell_size=10.0, n_atoms=2, atomic_numbers=[1, 1])
    dyn_config = DynamicsConfig(selection_rate=0.5)

    assert isinstance(ComponentFactory.get_generator(gen_config), MockGenerator)
    assert isinstance(ComponentFactory.get_oracle(OracleConfig()), MockOracle)
    assert isinstance(ComponentFactory.get_trainer(TrainerConfig()), MockTrainer)
    assert isinstance(ComponentFactory.get_dynamics(dyn_config), MockDynamics)
    assert isinstance(ComponentFactory.get_validator(ValidatorConfig()), MockValidator)


def test_factory_invalid_role() -> None:
    # ComponentFactory.create expects a ComponentConfig.
    # It checks role first.
    with pytest.raises(ValueError, match="Unknown component role"):
        ComponentFactory.create("unknown_role", ComponentConfig(name="mock"))


def test_factory_invalid_type() -> None:
    # We need to bypass validation to test invalid type string inside factory logic
    # But ComponentConfig enforces name literals if we use specific Configs.
    # However, create accepts ComponentConfig which allows any name string (base class).

    # If I use a name not in registry but valid in Config:
    with pytest.raises(ValueError, match="Unknown component type"):
        ComponentFactory.get_generator(GeneratorConfig(name="random", cell_size=10.0, n_atoms=2, atomic_numbers=[1, 1]))


def test_mock_generator_iterator() -> None:
    config = GeneratorConfig(cell_size=10.0, n_atoms=2, atomic_numbers=[1, 1])
    generator = MockGenerator(config)
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
