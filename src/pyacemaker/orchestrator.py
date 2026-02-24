import json
import logging
from pathlib import Path

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.dataset import DatasetManager
from pyacemaker.core.factory import ComponentFactory
from pyacemaker.domain_models.state import PipelineState
from pyacemaker.modules.active_learner import ActiveLearner
from pyacemaker.modules.mace_workflow import MaceDistillationWorkflow
from pyacemaker.modules.oracle import MaceSurrogateOracle
from pyacemaker.modules.structure_generator import StructureGenerator
from pyacemaker.modules.trainer import PacemakerTrainer

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Main orchestrator for the PyAceMaker pipeline.
    Manages the execution flow, state persistence, and component coordination.
    """

    def __init__(self, config: PYACEMAKERConfig, base_dir: Path):
        self.config = config
        self.base_dir = base_dir
        self.dataset_manager = DatasetManager(config.dataset)

        # Initialize components using Factory
        self.factory = ComponentFactory(config)
        self.oracle = self.factory.create_oracle()
        self.validator = self.factory.create_validator()
        self.trainer = self.factory.create_trainer()

        # State management
        self.state_file = self.base_dir / "pipeline_state.json"
        self.state = self._load_state()

        # Components for Distillation (lazy loaded or init here)
        # We initialize them here for simplicity, but in a real app might be lazy
        self.structure_generator = StructureGenerator(config.structure_generator)
        self.active_learner = ActiveLearner(config.active_learning, self.oracle) # Needs logic fix if circular

        # MACE specific (mock/real)
        # In a real scenario, we might use a factory for these too
        from pyacemaker.trainer.mace_trainer import MaceTrainer
        self.mace_trainer = MaceTrainer(config.trainer) # Re-using trainer config or specific?

        self.mace_oracle = MaceSurrogateOracle(config.oracle) # Config might differ
        self.pacemaker_trainer = PacemakerTrainer(config.trainer)


    def _load_state(self) -> PipelineState:
        """Loads pipeline state from disk or creates new."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                return PipelineState(**data)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}. Starting fresh.")

        return PipelineState(
            current_step=1,
            total_steps=8 if self.config.distillation.enabled else 4,
            artifacts={}
        )

    def _save_state(self):
        """Persists pipeline state to disk."""
        with open(self.state_file, "w") as f:
            f.write(self.state.model_dump_json(indent=2))

    def _create_mace_workflow(self) -> MaceDistillationWorkflow:
        """Creates the MaceDistillationWorkflow instance with dependencies."""
        return MaceDistillationWorkflow(
            config=self.config.distillation,
            dataset_manager=self.dataset_manager,
            active_learner=self.active_learner,
            structure_generator=self.structure_generator,
            oracle=self.oracle,
            mace_oracle=self.mace_oracle,
            pacemaker_trainer=self.pacemaker_trainer,
            mace_trainer=self.mace_trainer,
            work_dir=self.base_dir / "distillation_work"
        )

    def run(self):
        """Executes the pipeline based on configuration."""
        logger.info("Starting PyAceMaker Pipeline")

        if self.config.distillation.enabled:
            self._run_mace_distillation()
        else:
            self._run_standard_pipeline()

        logger.info("Pipeline completed successfully")

    def _run_standard_pipeline(self):
        """Legacy/Standard pipeline execution."""
        # ... (implementation of standard pipeline)

    def _run_mace_distillation(self):
        """Executes the MACE Distillation Workflow."""
        workflow = self._create_mace_workflow()

        steps = [
            (1, workflow.step1_direct_sampling),
            (2, workflow.step2_active_learning_loop),
            (3, workflow.step3_final_mace_training),
            (4, workflow.step4_surrogate_data_generation),
            (5, workflow.step5_surrogate_labeling),
            (6, workflow.step6_pacemaker_base_training),
            (7, workflow.step7_delta_learning),
        ]

        for step_num, step_func in steps:
            if self.state.current_step <= step_num:
                logger.info(f"Executing Step {step_num}")
                try:
                    self.state = step_func(self.state)
                    self._save_state()
                except Exception as e:
                    logger.error(f"Step {step_num} failed: {e}")
                    raise
            else:
                logger.info(f"Skipping Step {step_num} (Already completed)")
