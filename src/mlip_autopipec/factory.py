from typing import Any, ClassVar

from mlip_autopipec.components.base import BaseComponent
from mlip_autopipec.components.mocks import (
    MockDynamics,
    MockGenerator,
    MockOracle,
    MockTrainer,
    MockValidator,
)
from mlip_autopipec.config import (
    BaseComponentConfig,
)
from mlip_autopipec.domain_models.enums import (
    ComponentRole,
    DynamicsType,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)


class ComponentFactory:
    """Factory for creating pipeline components."""

    _REGISTRY: ClassVar[dict[Any, dict[Any, type[BaseComponent]]]] = {
        ComponentRole.GENERATOR: {
            GeneratorType.RANDOM: MockGenerator,  # Placeholder for real implementations
            GeneratorType.ADAPTIVE: MockGenerator,
        },
        ComponentRole.ORACLE: {
            OracleType.QE: MockOracle,
            OracleType.VASP: MockOracle,
            OracleType.MOCK: MockOracle,
        },
        ComponentRole.TRAINER: {
            TrainerType.PACEMAKER: MockTrainer,
            TrainerType.MOCK: MockTrainer,
        },
        ComponentRole.DYNAMICS: {
            DynamicsType.LAMMPS: MockDynamics,
            DynamicsType.EON: MockDynamics,
            DynamicsType.MOCK: MockDynamics,
        },
        ComponentRole.VALIDATOR: {
            ValidatorType.STANDARD: MockValidator,
            ValidatorType.MOCK: MockValidator,
        },
    }

    @classmethod
    def create(cls, role: ComponentRole, config: BaseComponentConfig) -> Any:
        """Create a component instance based on role and config."""
        if role not in cls._REGISTRY:
            msg = f"Unknown component role: {role}"
            raise ValueError(msg)

        # Determine the specific type from config
        # We rely on the config having a 'type' field that matches the enum
        if not hasattr(config, "type"):
            msg = f"Config for {role} must have a 'type' field"
            raise ValueError(msg)

        component_type_enum = config.type

        type_registry = cls._REGISTRY[role]

        if component_type_enum not in type_registry:
            msg = f"Unknown component type: {component_type_enum} for role {role}"
            raise ValueError(msg)

        component_class = type_registry[component_type_enum]

        return component_class(config)
