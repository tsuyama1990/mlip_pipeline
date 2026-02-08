from typing import Any, ClassVar

from mlip_autopipec.components.dynamics import BaseDynamics, MockDynamics
from mlip_autopipec.components.generator import BaseGenerator, MockGenerator
from mlip_autopipec.components.oracle import BaseOracle, MockOracle
from mlip_autopipec.components.trainer import BaseTrainer, MockTrainer
from mlip_autopipec.components.validator import BaseValidator, MockValidator
from mlip_autopipec.interfaces.base_component import BaseComponent


class ComponentFactory:
    _REGISTRY: ClassVar[dict[str, dict[str, type[BaseComponent]]]] = {
        "generator": {"mock": MockGenerator},
        "oracle": {"mock": MockOracle},
        "trainer": {"mock": MockTrainer},
        "dynamics": {"mock": MockDynamics},
        "validator": {"mock": MockValidator},
    }

    @classmethod
    def create(cls, component_role: str, config: dict[str, Any]) -> BaseComponent:
        component_type = config.get("type", "mock")
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
    def get_generator(cls, config: dict[str, Any]) -> BaseGenerator:
        return cls.create("generator", config)  # type: ignore

    @classmethod
    def get_oracle(cls, config: dict[str, Any]) -> BaseOracle:
        return cls.create("oracle", config)  # type: ignore

    @classmethod
    def get_trainer(cls, config: dict[str, Any]) -> BaseTrainer:
        return cls.create("trainer", config)  # type: ignore

    @classmethod
    def get_dynamics(cls, config: dict[str, Any]) -> BaseDynamics:
        return cls.create("dynamics", config)  # type: ignore

    @classmethod
    def get_validator(cls, config: dict[str, Any]) -> BaseValidator:
        return cls.create("validator", config)  # type: ignore
