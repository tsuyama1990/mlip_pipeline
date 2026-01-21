import logging
from pathlib import Path
from typing import TYPE_CHECKING

from mlip_autopipec.dft.runner import QERunner
from mlip_autopipec.generator.builder import StructureBuilder
from mlip_autopipec.orchestration.interfaces import BuilderProtocol, SurrogateProtocol
from mlip_autopipec.surrogate.pipeline import SurrogatePipeline
from mlip_autopipec.training.config_gen import TrainConfigGenerator
from mlip_autopipec.training.dataset import DatasetBuilder
from mlip_autopipec.training.pacemaker import PacemakerWrapper

if TYPE_CHECKING:
    from mlip_autopipec.orchestration.manager import WorkflowManager

logger = logging.getLogger(__name__)


class PhaseExecutor:
    """
    Handles the execution of individual workflow phases to decouple logic from WorkflowManager.
    """

    def __init__(self, manager: "WorkflowManager") -> None:
        self.manager = manager
        self.config = manager.config
        self.db = manager.db_manager
        self.queue = manager.task_queue

        # Lazy dependencies
        self._builder: BuilderProtocol | None = manager.builder
        self._surrogate: SurrogateProtocol | None = manager.surrogate

    def execute_exploration(self) -> None:
        """Execute Phase A: Exploration."""
        logger.info("Phase A: Exploration")
        try:
            if not self._builder:
                self._builder = StructureBuilder(self.config)

            candidates = self._builder.build()
            logger.info(f"Generated {len(candidates)} raw candidates.")

            if self.config.surrogate_config:
                if not self._surrogate:
                    self._surrogate = SurrogatePipeline(self.config.surrogate_config)

                selected, _ = self._surrogate.run(candidates)
                logger.info(f"Selected {len(selected)} candidates via Surrogate.")
            else:
                selected = candidates
                logger.info("Surrogate skipped (no config). Using all candidates.")

            for atoms in selected:
                self.db.save_candidate(
                    atoms,
                    {"status": "pending", "generation": self.manager.state.current_generation},
                )

        except Exception:
            logger.exception("Exploration phase failed")

    def execute_dft(self) -> None:
        """Execute Phase B: DFT Labeling."""
        logger.info("Phase B: DFT Labeling")
        try:
            atoms_list = self.db.get_atoms("status=pending")
            if not atoms_list:
                logger.warning("No pending atoms found for DFT.")
                return

            logger.info(f"Found {len(atoms_list)} pending atoms for DFT.")
            if self.config.dft_config:
                runner = QERunner(self.config.dft_config)
                futures = self.queue.submit_dft_batch(runner.run, atoms_list)
                results = self.queue.wait_for_completion(futures)

                success_count = 0
                for atoms, res in zip(atoms_list, results, strict=True):
                    if res:
                        self.db.save_dft_result(
                            atoms,
                            res,
                            {
                                "status": "training",
                                "generation": self.manager.state.current_generation,
                            },
                        )
                        success_count += 1
                    else:
                        logger.warning("DFT failed for an atom.")

                logger.info(f"DFT Phase complete. Success: {success_count}/{len(atoms_list)}")

        except Exception:
            logger.exception("DFT phase failed")

    def execute_training(self) -> None:
        """Execute Phase C: Training."""
        logger.info("Phase C: Training")
        try:
            if not self.config.training_config:
                logger.warning("No Training Config. Skipping training.")
                return

            dataset_builder = DatasetBuilder(self.db)
            template_path = self.config.training_config.template_file or Path("input.yaml")
            config_gen = TrainConfigGenerator(template_path=template_path)
            wrapper = PacemakerWrapper()

            result = wrapper.train(
                self.config.training_config,
                dataset_builder,
                config_gen,
                self.manager.work_dir,
                self.manager.state.current_generation,
            )
            logger.info(f"Training complete. Potential at: {result.potential_path}")

        except Exception:
            logger.exception("Training phase failed")

    def execute_inference(self) -> None:
        """Execute Phase D: Inference."""
        logger.info("Phase D: Inference")
        try:
            if not self.config.inference_config:
                logger.warning("No Inference Config. Skipping inference.")
            else:
                # Logic for inference execution (placeholder as per instruction)
                pass
        except Exception:
            logger.exception("Inference phase failed")
