import logging
import shutil

from mlip_autopipec.orchestration.phases.base import BasePhase
from mlip_autopipec.training.dataset import DatasetBuilder
from mlip_autopipec.training.pacemaker import PacemakerWrapper

logger = logging.getLogger(__name__)


class TrainingPhase(BasePhase):
    def execute(self) -> None:
        """Execute Phase C: Training."""
        logger.info("Phase C: Training")
        try:
            if not self.config.training_config:
                logger.warning("No Training Config. Skipping training.")
                return

            dataset_builder = DatasetBuilder(self.db)

            logger.info("Exporting training data...")
            dataset_builder.export(self.config.training_config, self.manager.work_dir)

            logger.info("Initializing Pacemaker...")
            wrapper = PacemakerWrapper(self.config.training_config, self.manager.work_dir)

            # Assuming previous generation's potential can be used as initial
            initial_potential = None
            prev_gen = self.manager.state.cycle_index - 1
            if prev_gen >= 0:
                prev_pot = self.manager.work_dir / "potentials" / f"generation_{prev_gen}.yace"
                if prev_pot.exists():
                    initial_potential = prev_pot

            logger.info(f"Starting training (Gen {self.manager.state.cycle_index})...")
            result = wrapper.train(initial_potential=initial_potential)

            if result.success and result.potential_path:
                logger.info(f"Training complete. Potential at: {result.potential_path}")
                # Save potential to generation specific path
                pot_dir = self.manager.work_dir / "potentials"
                pot_dir.mkdir(exist_ok=True)
                dest = pot_dir / f"generation_{self.manager.state.cycle_index}.yace"

                # Update state
                self.manager.state.latest_potential_path = dest

                # Copy or move
                try:
                    shutil.copy2(result.potential_path, dest)
                    # Also update 'current.yace' link/copy
                    current = self.manager.work_dir / "current.yace"
                    shutil.copy2(result.potential_path, current)
                except Exception:
                    logger.exception("Failed to save potential artifacts")
            else:
                logger.error("Training failed.")

        except Exception:
            logger.exception("Training phase failed")
