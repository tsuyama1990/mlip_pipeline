import pytest

from mlip_autopipec.components.dynamics import EONDynamics, LAMMPSDynamics, MockDynamics
from mlip_autopipec.components.generator import AdaptiveGenerator, MockGenerator
from mlip_autopipec.components.oracle import MockOracle, QEOracle
from mlip_autopipec.components.trainer import MockTrainer, PacemakerTrainer
from mlip_autopipec.components.validator import MockValidator, StandardValidator
from mlip_autopipec.domain_models.config import (
    AdaptiveGeneratorConfig,
    ComponentConfig,
    EONDynamicsConfig,
    LAMMPSDynamicsConfig,
    MockDynamicsConfig,
    MockGeneratorConfig,
    MockOracleConfig,
    MockTrainerConfig,
    MockValidatorConfig,
    PacemakerTrainerConfig,
    QEOracleConfig,
    StandardValidatorConfig,
)
from mlip_autopipec.domain_models.enums import (
    ComponentRole,
    DynamicsType,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)
from mlip_autopipec.factory import ComponentFactory


def test_factory_creation_mock() -> None:
    # Use valid configs for factory creation
    gen_config = MockGeneratorConfig(
        name=GeneratorType.MOCK, cell_size=10.0, n_atoms=2, atomic_numbers=[1, 1]
    )
    dyn_config = MockDynamicsConfig(
        name=DynamicsType.MOCK, selection_rate=0.5, uncertainty_threshold=5.0
    )
    oracle_config = MockOracleConfig(name=OracleType.MOCK)
    trainer_config = MockTrainerConfig(name=TrainerType.MOCK)
    validator_config = MockValidatorConfig(name=ValidatorType.MOCK)

    assert isinstance(ComponentFactory.get_generator(gen_config), MockGenerator)
    assert isinstance(ComponentFactory.get_oracle(oracle_config), MockOracle)
    assert isinstance(ComponentFactory.get_trainer(trainer_config), MockTrainer)
    assert isinstance(ComponentFactory.get_dynamics(dyn_config), MockDynamics)
    assert isinstance(ComponentFactory.get_validator(validator_config), MockValidator)


def test_factory_creation_real() -> None:
    # Test creation of real (placeholder) components
    oracle_config = QEOracleConfig(
        name=OracleType.QE, kspacing=0.04, pseudopotentials={}, ecutwfc=60.0, ecutrho=360.0
    )
    trainer_config = PacemakerTrainerConfig(
        name=TrainerType.PACEMAKER, max_num_epochs=10, basis_size=500, cutoff=4.0
    )
    dyn_config = LAMMPSDynamicsConfig(name=DynamicsType.LAMMPS, timestep=0.001, n_steps=100)
    eon_config = EONDynamicsConfig(name=DynamicsType.EON, temperature=400.0)
    validator_config = StandardValidatorConfig(name=ValidatorType.STANDARD)
    # Adaptive generator requires a lot of fields
    gen_config = AdaptiveGeneratorConfig(
        name=GeneratorType.ADAPTIVE, element="Cu", crystal_structure="fcc", n_structures=5
    )

    assert isinstance(ComponentFactory.get_oracle(oracle_config), QEOracle)
    assert isinstance(ComponentFactory.get_trainer(trainer_config), PacemakerTrainer)
    assert isinstance(ComponentFactory.get_dynamics(dyn_config), LAMMPSDynamics)
    assert isinstance(ComponentFactory.get_dynamics(eon_config), EONDynamics)
    assert isinstance(ComponentFactory.get_validator(validator_config), StandardValidator)
    assert isinstance(ComponentFactory.get_generator(gen_config), AdaptiveGenerator)


def test_factory_invalid_role() -> None:
    # ComponentFactory.create expects a ComponentConfig.
    # It checks role first.
    # Note: Type checker catches this, so we use type: ignore to test runtime check
    with pytest.raises(ValueError, match="Unknown component role"):
        ComponentFactory.create("unknown_role", ComponentConfig(name="mock"))  # type: ignore


def test_factory_invalid_type() -> None:
    # We can pass a generic ComponentConfig with an invalid name
    config = ComponentConfig(name="random_invalid_type")
    with pytest.raises(ValueError, match="Unknown component type"):
        ComponentFactory.create(ComponentRole.GENERATOR, config)


def test_mock_generator_iterator() -> None:
    config = MockGeneratorConfig(
        name=GeneratorType.MOCK, cell_size=10.0, n_atoms=2, atomic_numbers=[1, 1]
    )
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
