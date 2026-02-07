from mlip_autopipec.constants import TYPE_MOCK
from mlip_autopipec.domain_models import (
    DynamicsConfig,
    GeneratorConfig,
    OracleConfig,
    SelectorConfig,
    TrainerConfig,
    ValidatorConfig,
)
from mlip_autopipec.infrastructure.mocks import (
    MockDynamics,
    MockOracle,
    MockSelector,
    MockStructureGenerator,
    MockTrainer,
    MockValidator,
)
from mlip_autopipec.interfaces import (
    BaseDynamics,
    BaseOracle,
    BaseSelector,
    BaseStructureGenerator,
    BaseTrainer,
    BaseValidator,
)


def create_oracle(config: OracleConfig) -> BaseOracle:
    if config.type == TYPE_MOCK:
        return MockOracle(params={"noise_level": config.noise_level})
    msg = f"Unknown oracle type: {config.type}"
    raise ValueError(msg)


def create_trainer(config: TrainerConfig) -> BaseTrainer:
    if config.type == TYPE_MOCK:
        return MockTrainer(params={})
    msg = f"Unknown trainer type: {config.type}"
    raise ValueError(msg)


def create_dynamics(config: DynamicsConfig) -> BaseDynamics:
    if config.type == TYPE_MOCK:
        return MockDynamics(params={"prob_halt": config.prob_halt})
    msg = f"Unknown dynamics type: {config.type}"
    raise ValueError(msg)


def create_generator(config: GeneratorConfig) -> BaseStructureGenerator:
    if config.type == TYPE_MOCK:
        return MockStructureGenerator(params={})
    msg = f"Unknown generator type: {config.type}"
    raise ValueError(msg)


def create_validator(config: ValidatorConfig) -> BaseValidator:
    if config.type == TYPE_MOCK:
        return MockValidator(params={})
    msg = f"Unknown validator type: {config.type}"
    raise ValueError(msg)


def create_selector(config: SelectorConfig) -> BaseSelector:
    if config.type == TYPE_MOCK:
        return MockSelector(params={})
    msg = f"Unknown selector type: {config.type}"
    raise ValueError(msg)
