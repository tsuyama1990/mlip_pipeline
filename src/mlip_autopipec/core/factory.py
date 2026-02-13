import logging
from pathlib import Path

from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.domain_models.enums import (
    DynamicsType,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)
from mlip_autopipec.dynamics import BaseDynamics, EONDriver, LAMMPSDriver, MockDynamics
from mlip_autopipec.generator import (
    AdaptiveGenerator,
    BaseGenerator,
    M3GNetGenerator,
    MockGenerator,
    RandomGenerator,
)
from mlip_autopipec.oracle import BaseOracle, DFTManager, MockOracle
from mlip_autopipec.trainer import BaseTrainer, MockTrainer, PacemakerTrainer
from mlip_autopipec.validator import BaseValidator, MockValidator
from mlip_autopipec.validator.physics import PhysicsValidator

logger = logging.getLogger(__name__)


class ComponentFactory:
    """Factory for creating pipeline components."""

    def __init__(self, config: GlobalConfig) -> None:
        self.config = config

    def create_generator(self) -> BaseGenerator:
        gen_type = self.config.generator.type
        if gen_type == GeneratorType.MOCK:
            return MockGenerator(self.config.generator)
        if gen_type == GeneratorType.RANDOM:
            return RandomGenerator(self.config.generator)
        if gen_type == GeneratorType.M3GNET:
            return M3GNetGenerator(self.config.generator)
        if gen_type == GeneratorType.ADAPTIVE:
            return AdaptiveGenerator(self.config.generator)

        msg = f"Unsupported generator type: {gen_type}"
        raise ValueError(msg)

    def create_oracle(self) -> BaseOracle:
        oracle_type = self.config.oracle.type
        if oracle_type == OracleType.MOCK:
            return MockOracle(self.config.oracle)
        if oracle_type == OracleType.DFT:
            return DFTManager(self.config.oracle)

        msg = f"Unsupported oracle type: {oracle_type}"
        raise ValueError(msg)

    def create_trainer(self, work_dir: Path) -> BaseTrainer:
        trainer_type = self.config.trainer.type
        if trainer_type == TrainerType.MOCK:
            return MockTrainer(work_dir)
        if trainer_type == TrainerType.PACEMAKER:
            return PacemakerTrainer(work_dir, self.config.trainer)

        msg = f"Unsupported trainer type: {trainer_type}"
        raise ValueError(msg)

    def create_dynamics(self, work_dir: Path) -> BaseDynamics:
        dyn_type = self.config.dynamics.type
        if dyn_type == DynamicsType.MOCK:
            return MockDynamics(self.config.dynamics)
        if dyn_type == DynamicsType.LAMMPS:
            return LAMMPSDriver(work_dir, self.config.dynamics)
        if dyn_type == DynamicsType.EON:
            return EONDriver(work_dir, self.config.dynamics)

        msg = f"Unsupported dynamics type: {dyn_type}"
        raise ValueError(msg)

    def create_validator(self) -> BaseValidator:
        val_type = self.config.validator.type
        if val_type == ValidatorType.MOCK:
            return MockValidator()
        if val_type == ValidatorType.PHYSICS:
            return PhysicsValidator(self.config)
        msg = f"Unsupported validator type: {val_type}"
        raise ValueError(msg)
