from mlip_autopipec.domain_models.config import (
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
from mlip_autopipec.infrastructure.mocks import (
    MockDynamics,
    MockGenerator,
    MockOracle,
    MockSelector,
    MockTrainer,
    MockValidator,
)


def test_create_oracle_mock() -> None:
    config = MockOracleConfig(noise_level=0.05)
    oracle = create_oracle(config)
    assert isinstance(oracle, MockOracle)
    assert oracle.params["noise_level"] == 0.05


def test_create_trainer_mock() -> None:
    config = MockTrainerConfig()
    trainer = create_trainer(config)
    assert isinstance(trainer, MockTrainer)


def test_create_dynamics_mock() -> None:
    config = MockDynamicsConfig()
    dyn = create_dynamics(config)
    assert isinstance(dyn, MockDynamics)


def test_create_generator_mock() -> None:
    config = MockGeneratorConfig()
    gen = create_generator(config)
    assert isinstance(gen, MockGenerator)


def test_create_validator_mock() -> None:
    config = MockValidatorConfig()
    val = create_validator(config)
    assert isinstance(val, MockValidator)


def test_create_selector_mock() -> None:
    config = MockSelectorConfig()
    sel = create_selector(config)
    assert isinstance(sel, MockSelector)
