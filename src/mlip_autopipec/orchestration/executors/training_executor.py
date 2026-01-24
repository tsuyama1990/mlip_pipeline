import logging
import shutil
from pathlib import Path

from mlip_autopipec.orchestration.executors.base_executor import BaseExecutor
from mlip_autopipec.training.dataset import DatasetBuilder
from mlip_autopipec.training.pacemaker import PacemakerWrapper
from mlip_autopipec.config.schemas.training import TrainingConfig

logger = logging.getLogger(__name__)

class TrainingExecutor(BaseExecutor):
    """Executes Phase C: Training."""

    def _create_pacemaker_wrapper(self, config: TrainingConfig, work_dir: Path) -> PacemakerWrapper:
        return PacemakerWrapper(config, work_dir)

    def execute(self) -> bool:
        """Execute Phase C: Training."""
        logger.info("Phase C: Training")
        try:
            if not self.config.training_config:
                logger.warning("No Training Config. Skipping training.")
                return False

            dataset_builder = DatasetBuilder(self.db)

            logger.info("Exporting training data...")
            dataset_builder.export(self.config.training_config, self.manager.work_dir)

            logger.info("Initializing Pacemaker...")
            wrapper = self._create_pacemaker_wrapper(self.config.training_config, self.manager.work_dir)

            # Assuming previous generation's potential can be used as initial
            initial_potential = None
            prev_gen = self.manager.state.current_generation - 1
            if prev_gen >= 0:
                 prev_pot = self.manager.work_dir / "potentials" / f"generation_{prev_gen}.yace"
                 if prev_pot.exists():
                     initial_potential = prev_pot

            logger.info(f"Starting training (Gen {self.manager.state.current_generation})...")
            result = wrapper.train(initial_potential=initial_potential)

            if result.success and result.potential_path:
                logger.info(f"Training complete. Potential at: {result.potential_path}")
                # Save potential to generation specific path
                pot_dir = self.manager.work_dir / "potentials"
                pot_dir.mkdir(exist_ok=True)
                dest = pot_dir / f"generation_{self.manager.state.current_generation}.yace"

                try:
                    shutil.copy2(result.potential_path, dest)
                    # Also update 'current.yace' link/copy
                    current = self.manager.work_dir / "current.yace"
                    shutil.copy2(result.potential_path, current)
                    return True
                except Exception:
                    logger.exception("Failed to save potential artifacts")
                    return False
            else:
                logger.error("Training failed.")
                return False

        except Exception:
            logger.exception("Training phase failed")
            return False
