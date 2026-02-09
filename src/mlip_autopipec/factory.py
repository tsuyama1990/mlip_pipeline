from typing import Any, ClassVar

from mlip_autopipec.components.dynamics import (
    BaseDynamics,
    EONDynamics,
    LAMMPSDynamics,
    MockDynamics,
)
from mlip_autopipec.components.generator import (
    AdaptiveGenerator,
    BaseGenerator,
    MockGenerator,
)
from mlip_autopipec.components.oracle import BaseOracle, MockOracle, QEOracle, VASPOracle
from mlip_autopipec.components.trainer import BaseTrainer, MockTrainer, PacemakerTrainer
from mlip_autopipec.components.validator import (
    BaseValidator,
    MockValidator,
    StandardValidator,
)
from mlip_autopipec.domain_models.config import (
    ComponentConfig,
    DynamicsConfig,
    GeneratorConfig,
    OracleConfig,
    TrainerConfig,
    ValidatorConfig,
)
from mlip_autopipec.domain_models.enums import (
    ComponentRole,
    DynamicsType,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)
from mlip_autopipec.interfaces.base_component import BaseComponent


class ComponentFactory:
    _REGISTRY: ClassVar[dict[ComponentRole, dict[str, type[BaseComponent[Any]]]]] = {
        ComponentRole.GENERATOR: {
            GeneratorType.MOCK: MockGenerator,
            GeneratorType.ADAPTIVE: AdaptiveGenerator,
        },
        ComponentRole.ORACLE: {
            OracleType.MOCK: MockOracle,
            OracleType.QE: QEOracle,
            OracleType.VASP: VASPOracle,
        },
        ComponentRole.TRAINER: {
            TrainerType.MOCK: MockTrainer,
            TrainerType.PACEMAKER: PacemakerTrainer,
        },
        ComponentRole.DYNAMICS: {
            DynamicsType.MOCK: MockDynamics,
            DynamicsType.LAMMPS: LAMMPSDynamics,
            DynamicsType.EON: EONDynamics,
        },
        ComponentRole.VALIDATOR: {
            ValidatorType.MOCK: MockValidator,
            ValidatorType.STANDARD: StandardValidator,
        },
    }

    @classmethod
    def create(cls, component_role: ComponentRole, config: ComponentConfig) -> BaseComponent[Any]:
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
        return cls.create(ComponentRole.GENERATOR, config)  # type: ignore

    @classmethod
    def get_oracle(cls, config: OracleConfig) -> BaseOracle:
        return cls.create(ComponentRole.ORACLE, config)  # type: ignore

    @classmethod
    def get_trainer(cls, config: TrainerConfig) -> BaseTrainer:
        return cls.create(ComponentRole.TRAINER, config)  # type: ignore

    @classmethod
    def get_dynamics(cls, config: DynamicsConfig) -> BaseDynamics:
        return cls.create(ComponentRole.DYNAMICS, config)  # type: ignore

    @classmethod
    def get_validator(cls, config: ValidatorConfig) -> BaseValidator:
        return cls.create(ComponentRole.VALIDATOR, config)  # type: ignore

    def __repr__(self) -> str:
        return "<ComponentFactory>"

    def __str__(self) -> str:
        return "ComponentFactory"
