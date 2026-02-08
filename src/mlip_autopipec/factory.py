from typing import Any, ClassVar

from mlip_autopipec.components.dynamics import BaseDynamics, MockDynamics
from mlip_autopipec.components.generator import (
    AdaptiveGenerator,
    BaseGenerator,
    MockGenerator,
)
from mlip_autopipec.components.oracle import BaseOracle, MockOracle
from mlip_autopipec.components.trainer import BaseTrainer, MockTrainer
from mlip_autopipec.components.validator import BaseValidator, MockValidator
from mlip_autopipec.domain_models.config import (
    ComponentConfig,
    DynamicsConfig,
    GeneratorConfig,
    OracleConfig,
    TrainerConfig,
    ValidatorConfig,
)
from mlip_autopipec.domain_models.enums import (
    DynamicsType,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)
from mlip_autopipec.interfaces.base_component import BaseComponent


class ComponentFactory:
    _REGISTRY: ClassVar[dict[str, dict[str, type[BaseComponent[Any]]]]] = {
        "generator": {
            GeneratorType.MOCK: MockGenerator,
            GeneratorType.ADAPTIVE: AdaptiveGenerator,
        },
        "oracle": {OracleType.MOCK: MockOracle},
        "trainer": {TrainerType.MOCK: MockTrainer},
        "dynamics": {DynamicsType.MOCK: MockDynamics},
        "validator": {ValidatorType.MOCK: MockValidator},
    }

    @classmethod
    def create(cls, component_role: str, config: ComponentConfig) -> BaseComponent[Any]:
        component_type = config.name
        if component_role not in cls._REGISTRY:
            msg = f"Unknown component role: {component_role}"
            raise ValueError(msg)

        type_registry = cls._REGISTRY[component_role]
        if component_type not in type_registry:
            msg = f"Unknown component type '{component_type}' for role '{component_role}'"
            raise ValueError(msg)

        component_class = type_registry[component_type]
        return component_class(config)

    @classmethod
    def get_generator(cls, config: GeneratorConfig) -> BaseGenerator:
        return cls.create("generator", config)  # type: ignore

    @classmethod
    def get_oracle(cls, config: OracleConfig) -> BaseOracle:
        return cls.create("oracle", config)  # type: ignore

    @classmethod
    def get_trainer(cls, config: TrainerConfig) -> BaseTrainer:
        return cls.create("trainer", config)  # type: ignore

    @classmethod
    def get_dynamics(cls, config: DynamicsConfig) -> BaseDynamics:
        return cls.create("dynamics", config)  # type: ignore

    @classmethod
    def get_validator(cls, config: ValidatorConfig) -> BaseValidator:
        return cls.create("validator", config)  # type: ignore
