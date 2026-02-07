from mlip_autopipec.domain_models.config import (
    DynamicsConfig,
    GeneratorConfig,
    MockDynamicsConfig,
    MockGeneratorConfig,
    MockOracleConfig,
    MockSelectorConfig,
    MockTrainerConfig,
    MockValidatorConfig,
    OracleConfig,
    SelectorConfig,
    TrainerConfig,
    ValidatorConfig,
)
from mlip_autopipec.infrastructure.mocks import (
    MockDynamics,
    MockGenerator,
    MockOracle,
    MockSelector,
    MockTrainer,
    MockValidator,
)
from mlip_autopipec.interfaces.dynamics import BaseDynamics
from mlip_autopipec.interfaces.generator import BaseGenerator
from mlip_autopipec.interfaces.oracle import BaseOracle
from mlip_autopipec.interfaces.selector import BaseSelector
from mlip_autopipec.interfaces.trainer import BaseTrainer
from mlip_autopipec.interfaces.validator import BaseValidator


def create_oracle(config: OracleConfig) -> BaseOracle:
    """
    Factory function to create an Oracle instance.
    """
    if isinstance(config, MockOracleConfig):
        return MockOracle(params=config.model_dump())
    # Cycle 01 only requires Mocks, but we prepare for others
    msg = f"Unknown Oracle type: {config}"
    raise NotImplementedError(msg)


def create_trainer(config: TrainerConfig) -> BaseTrainer:
    """
    Factory function to create a Trainer instance.
    """
    if isinstance(config, MockTrainerConfig):
        return MockTrainer(params=config.model_dump())
    msg = f"Unknown Trainer type: {config}"
    raise NotImplementedError(msg)


def create_dynamics(config: DynamicsConfig) -> BaseDynamics:
    """
    Factory function to create a Dynamics instance.
    """
    if isinstance(config, MockDynamicsConfig):
        return MockDynamics(params=config.model_dump())
    msg = f"Unknown Dynamics type: {config}"
    raise NotImplementedError(msg)


def create_generator(config: GeneratorConfig) -> BaseGenerator:
    """
    Factory function to create a Generator instance.
    """
    if isinstance(config, MockGeneratorConfig):
        return MockGenerator(params=config.model_dump())
    msg = f"Unknown Generator type: {config}"
    raise NotImplementedError(msg)


def create_validator(config: ValidatorConfig) -> BaseValidator:
    """
    Factory function to create a Validator instance.
    """
    if isinstance(config, MockValidatorConfig):
        return MockValidator(params=config.model_dump())
    msg = f"Unknown Validator type: {config}"
    raise NotImplementedError(msg)


def create_selector(config: SelectorConfig) -> BaseSelector:
    """
    Factory function to create a Selector instance.
    """
    if isinstance(config, MockSelectorConfig):
        return MockSelector(params=config.model_dump())
    msg = f"Unknown Selector type: {config}"
    raise NotImplementedError(msg)
