"""Orchestrator module implementation."""

from typing import TypeVar

from loguru import logger

from pyacemaker.core.base import BaseModule, ModuleResult
from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig
from pyacemaker.core.interfaces import (
    CycleResult,
    DynamicsEngine,
    IOrchestrator,
    Oracle,
    StructureGenerator,
    Trainer,
    UncertaintyModel,
    Validator,
)
from pyacemaker.core.utils import (
    atoms_to_metadata,
    metadata_to_atoms,
)
from pyacemaker.domain_models.models import (
    StructureMetadata,
)
from pyacemaker.modules.active_learner import ActiveLearner
from pyacemaker.modules.dynamics_engine import EONEngine, LAMMPSEngine
from pyacemaker.modules.oracle import DFTOracle, MaceSurrogateOracle, MockOracle
from pyacemaker.modules.structure_generator import (
    AdaptiveStructureGenerator,
    RandomStructureGenerator,
)
from pyacemaker.modules.trainer import MaceTrainer, PacemakerTrainer
from pyacemaker.modules.validator import Validator as DefaultValidator
from pyacemaker.oracle.dataset import DatasetManager
from pyacemaker.workflows.active_learning import StandardActiveLearningWorkflow
from pyacemaker.workflows.distillation import MaceDistillationWorkflow

# Import new DirectGenerator
try:
    from pyacemaker.generator.direct import DirectGenerator
    HAS_DIRECT = True
except ImportError:
    HAS_DIRECT = False

T = TypeVar("T", bound=BaseModule)


def _create_default_module(module_class: type[T], config: PYACEMAKERConfig) -> T:
    """Factory to create a default module instance."""
    return module_class(config)


class Orchestrator(IOrchestrator):
    """Main Orchestrator for the active learning cycle."""

    def __init__(
        self,
        config: PYACEMAKERConfig,
        structure_generator: StructureGenerator | None = None,
        oracle: Oracle | None = None,
        trainer: Trainer | None = None,
        dynamics_engine: DynamicsEngine | None = None,
        validator: Validator | None = None,
        mace_trainer: Trainer | None = None,
        active_learner: ActiveLearner | None = None,
    ) -> None:
        """Initialize the orchestrator and sub-modules."""
        super().__init__(config)
        self.config = config
        self.logger = logger.bind(name="Orchestrator")

        self.structure_generator = self._init_generator(structure_generator)
        self.oracle = self._init_oracle(oracle)
        self.uncertainty_model = self._init_uncertainty_model()
        self.trainer = trainer or _create_default_module(PacemakerTrainer, config)
        self.mace_trainer = self._init_mace_trainer(mace_trainer)
        self.active_learner = self._init_active_learner(active_learner)
        self.dynamics_engine = self._init_dynamics_engine(dynamics_engine)
        self.validator = self._init_validator(validator)

        self.dataset_manager = DatasetManager()

        # Initialize Workflows
        self.standard_workflow = StandardActiveLearningWorkflow(
            config,
            self.dataset_manager,
            self.structure_generator,
            self.oracle,
            self.trainer,
            self.dynamics_engine,
            self.validator,
        )

        self.distillation_workflow = self._init_distillation_workflow()

    def _init_generator(self, override: StructureGenerator | None) -> StructureGenerator:
        if override:
            return override
        if self.config.distillation.enable_mace_distillation and HAS_DIRECT:
            return _create_default_module(DirectGenerator, self.config)

        sg_cls = (
            AdaptiveStructureGenerator
            if self.config.structure_generator.strategy == "adaptive"
            else RandomStructureGenerator
        )
        return _create_default_module(sg_cls, self.config)

    def _init_oracle(self, override: Oracle | None) -> Oracle:
        if override:
            return override

        if self.config.distillation.enable_mace_distillation:
            oracle_cls = MockOracle if self.config.oracle.mock else DFTOracle
        elif self.config.oracle.mace:
            oracle_cls = MaceSurrogateOracle
        elif self.config.oracle.mock:
            oracle_cls = MockOracle
        else:
            oracle_cls = DFTOracle
        return _create_default_module(oracle_cls, self.config)

    def _init_uncertainty_model(self) -> UncertaintyModel | None:
        if self.config.distillation.enable_mace_distillation and self.config.oracle.mace:
             return _create_default_module(MaceSurrogateOracle, self.config)
        if isinstance(self.oracle, UncertaintyModel):
             return self.oracle
        return None

    def _init_mace_trainer(self, override: Trainer | None) -> Trainer | None:
        if override:
            return override
        if self.config.distillation.enable_mace_distillation:
            return _create_default_module(MaceTrainer, self.config)
        return None

    def _init_active_learner(self, override: ActiveLearner | None) -> ActiveLearner | None:
        if override:
            return override
        if self.config.distillation.enable_mace_distillation:
             return _create_default_module(ActiveLearner, self.config)
        return None

    def _init_dynamics_engine(self, override: DynamicsEngine | None) -> DynamicsEngine:
        if override:
            return override
        engine_cls: type[DynamicsEngine] = LAMMPSEngine
        if self.config.dynamics_engine.engine == "eon":
            engine_cls = EONEngine
        return _create_default_module(engine_cls, self.config)

    def _init_validator(self, override: Validator | None) -> Validator:
        if override:
            return override
        val_cls: type[Validator] = DefaultValidator
        if self.config.oracle.mock:
            from pyacemaker.modules.validator import MockValidator
            val_cls = MockValidator
        return _create_default_module(val_cls, self.config)

    def _init_distillation_workflow(self) -> MaceDistillationWorkflow | None:
        if not self.config.distillation.enable_mace_distillation:
            return None

        if not self.mace_trainer or not self.active_learner or not self.uncertainty_model:
            msg = "Missing components for MACE Distillation"
            raise RuntimeError(msg)

        return MaceDistillationWorkflow(
            self.config,
            self.dataset_manager,
            self.structure_generator,
            self.oracle,
            self.mace_trainer,
            self.active_learner,
            self.uncertainty_model,
            self.dynamics_engine,
            self.trainer,
        )

    def run(self) -> ModuleResult:
        """Run the active learning pipeline."""
        self.logger.info("Starting Active Learning Pipeline")

        if self.config.distillation.enable_mace_distillation:
            self.logger.info("Mode: MACE Distillation Workflow")
            if not self.distillation_workflow:
                msg = "Distillation Workflow not initialized"
                raise RuntimeError(msg)
            return self.distillation_workflow.run()

        self.logger.info("Mode: Standard Active Learning Loop")
        return self.standard_workflow.run()

    def run_step1_direct_sampling(self) -> list[StructureMetadata]:
        """Run Step 1: Direct Sampling (Public Interface for UAT)."""
        if not self.distillation_workflow:
             msg = "Distillation workflow not active"
             raise RuntimeError(msg)

        pool_path = self.distillation_workflow._step1_direct_sampling(self.config.distillation)
        return [
            atoms_to_metadata(a) for a in self.dataset_manager.load_iter(pool_path)
        ]

    def run_step2_active_learning(self) -> list[StructureMetadata]:
        """Run Step 2: Active Learning (Public Interface for UAT)."""
        pool_path = self.config.project.root_dir / "data" / "pool_structures.pckl.gzip"
        candidates_file = self.config.project.root_dir / CONSTANTS.default_candidates_file
        if not pool_path.exists() and candidates_file.exists():
            pool_path = candidates_file

        pool_iter = (
            atoms_to_metadata(a)
            for a in self.dataset_manager.load_iter(pool_path)
        )

        if not self.uncertainty_model:
            msg = "Uncertainty Model not initialized"
            raise RuntimeError(msg)

        # Streaming
        scored_pool = self.uncertainty_model.compute_uncertainty(pool_iter)

        if not self.active_learner:
            msg = "ActiveLearner not initialized"
            raise RuntimeError(msg)

        # Consumes stream
        selected = list(self.active_learner.select_batch(scored_pool))

        labeled = list(self.oracle.compute_batch(selected))

        atoms_stream = (metadata_to_atoms(s) for s in labeled)
        dataset_path = self.config.project.root_dir / "data" / self.config.orchestrator.dataset_file
        self.dataset_manager.save_iter(
            atoms_stream, dataset_path, mode="ab", calculate_checksum=False
        )

        return labeled

    def run_cycle(self) -> CycleResult:
        """Execute one active learning cycle (Standard Loop)."""
        return self.standard_workflow.run_cycle()
