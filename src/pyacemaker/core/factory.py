"""Module Factory Implementation."""

from typing import TYPE_CHECKING

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.interfaces import (
    DynamicsEngine,
    Oracle,
    StructureGenerator,
    Trainer,
    Validator,
)
from pyacemaker.modules.dynamics_engine import (
    ASEDynamicsEngine,
    EONEngine,
    LAMMPSEngine,
)
from pyacemaker.modules.oracle import DFTOracle, MaceSurrogateOracle, MockOracle
from pyacemaker.modules.structure_generator import (
    AdaptiveStructureGenerator,
    RandomStructureGenerator,
)
from pyacemaker.modules.trainer import MaceTrainer, PacemakerTrainer
from pyacemaker.modules.validator import Validator as ProductionValidator

if TYPE_CHECKING:
    pass


class ModuleFactory:
    """Factory for creating module instances based on configuration."""

    @staticmethod
    def create_structure_generator(config: PYACEMAKERConfig) -> StructureGenerator:
        """Create a structure generator instance."""
        if config.structure_generator.strategy == "adaptive":
            return AdaptiveStructureGenerator(config)
        return RandomStructureGenerator(config)

    @staticmethod
    def create_oracle(config: PYACEMAKERConfig) -> Oracle:
        """Create an oracle instance."""
        if config.oracle.mace:
            return MaceSurrogateOracle(config)
        if config.oracle.mock:
            return MockOracle(config)
        return DFTOracle(config)

    @staticmethod
    def create_trainer(config: PYACEMAKERConfig) -> Trainer:
        """Create a trainer instance (default: Pacemaker)."""
        return PacemakerTrainer(config)

    @staticmethod
    def create_mace_trainer(config: PYACEMAKERConfig) -> Trainer:
        """Create a MACE trainer instance."""
        return MaceTrainer(config)

    @staticmethod
    def create_dynamics_engine(config: PYACEMAKERConfig) -> DynamicsEngine:
        """Create a dynamics engine instance."""
        if config.dynamics_engine.engine == "eon":
            return EONEngine(config)
        if config.dynamics_engine.engine == "ase":
            return ASEDynamicsEngine(config)
        return LAMMPSEngine(config)

    @staticmethod
    def create_validator(config: PYACEMAKERConfig) -> Validator:
        """Create a validator instance."""
        if config.oracle.mock:
            # Lazy import to avoid potential circular dependencies if any
            # and because MockValidator is specifically for testing contexts
            from pyacemaker.modules.validator import MockValidator

            return MockValidator(config)
        return ProductionValidator(config)
