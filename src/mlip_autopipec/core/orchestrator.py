import logging

from ase.io import write

from mlip_autopipec.components.base import BaseGenerator, BaseOracle, BaseTrainer
from mlip_autopipec.components.mock import MockGenerator, MockOracle, MockTrainer
from mlip_autopipec.domain_models import (
    Config,
    Dataset,
    GeneratorType,
    OracleType,
    TrainerType,
    WorkflowStage,
)

from .state_manager import StateManager

logger = logging.getLogger("mlip_autopipec")

class Orchestrator:
    """
    Central controller for the active learning loop.
    """
    def __init__(self, config: Config) -> None:
        self.config = config
        self.work_dir = config.orchestrator.work_dir
        self.state_manager = StateManager(self.work_dir)
        self.state = self.state_manager.load()

        self.generator = self._init_generator()
        self.oracle = self._init_oracle()
        self.trainer = self._init_trainer()

    def _init_generator(self) -> BaseGenerator:
        if self.config.generator.type == GeneratorType.MOCK:
            return MockGenerator(self.config.generator)
        msg = f"Generator type {self.config.generator.type} not implemented"
        raise NotImplementedError(msg)

    def _init_oracle(self) -> BaseOracle:
        if self.config.oracle.type == OracleType.MOCK:
            return MockOracle(self.config.oracle)
        msg = f"Oracle type {self.config.oracle.type} not implemented"
        raise NotImplementedError(msg)

    def _init_trainer(self) -> BaseTrainer:
        if self.config.trainer.type == TrainerType.MOCK:
            return MockTrainer(self.config.trainer)
        msg = f"Trainer type {self.config.trainer.type} not implemented"
        raise NotImplementedError(msg)

    def run_loop(self) -> None:
        """
        Executes the main active learning loop.
        """
        logger.info(f"Starting orchestration loop. Max iterations: {self.config.orchestrator.max_iterations}")

        # If we are resuming, we should check if we already finished
        if self.state.iteration >= self.config.orchestrator.max_iterations:
             logger.info("Workflow already completed.")
             return

        while self.state.iteration < self.config.orchestrator.max_iterations:
            self.state.iteration += 1
            logger.info(f"--- Cycle {self.state.iteration}/{self.config.orchestrator.max_iterations} ---")

            # 1. Explore
            logger.info("Cycle %d: Exploring...", self.state.iteration)
            self.state.current_stage = WorkflowStage.EXPLORE
            self.state_manager.save(self.state)

            # For Cycle 01, we just generate N structures.
            structures_iter = self.generator.generate(n_structures=5, cycle=self.state.iteration)

            # 2. Label
            logger.info("Cycle %d: Labeling...", self.state.iteration)
            self.state.current_stage = WorkflowStage.LABEL
            self.state_manager.save(self.state)

            labelled_iter = self.oracle.compute(structures_iter)

            # Consume iterator and save to dataset
            dataset_path = self.work_dir / f"iteration_{self.state.iteration}" / "train.xyz"
            if not dataset_path.parent.exists():
                dataset_path.parent.mkdir(parents=True, exist_ok=True)

            # Materialize to list to write to file
            labelled_structures = list(labelled_iter)
            if not labelled_structures:
                 logger.warning("No structures generated/labeled in this cycle.")
                 # Should we continue or break? Continuing might lead to infinite loop if generator is broken.
                 # But maybe next cycle works.
                 continue

            # Convert Structure to Atoms for ASE write
            atoms_list = [s.atoms for s in labelled_structures]
            write(dataset_path, atoms_list)

            dataset = Dataset(path=dataset_path)

            # 3. Train
            logger.info("Cycle %d: Training...", self.state.iteration)
            self.state.current_stage = WorkflowStage.TRAIN
            self.state_manager.save(self.state)

            potential = self.trainer.train(dataset)
            self.state.latest_potential_path = potential.path

            self.state_manager.save(self.state)
            logger.info(f"Cycle {self.state.iteration} completed.")

        logger.info("Max iterations reached. Workflow finished.")
