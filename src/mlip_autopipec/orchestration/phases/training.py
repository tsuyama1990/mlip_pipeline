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

            cycle = self.manager.state.cycle_index
            training_dir = self.manager.work_dir / f"training_gen_{cycle}"
            training_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Training directory: {training_dir}")

            dataset_builder = DatasetBuilder(self.db)

            logger.info("Exporting training data...")
            # Export to training_dir. 'train.xyz' will be created in training_dir.
            output_path = training_dir / "dataset_marker"
            dataset_builder.export(output_path)

            logger.info("Initializing Pacemaker...")
            wrapper = PacemakerWrapper(self.config.training_config, training_dir)

            # Assuming previous generation's potential can be used as initial
            initial_potential = None
            if cycle > 0:
                prev_gen = cycle - 1
                pot_dir = self.manager.work_dir / "potentials"
                prev_pot = pot_dir / f"generation_{prev_gen}.yace"
                if prev_pot.exists():
                    initial_potential = prev_pot
                    logger.info(f"Using initial potential: {initial_potential}")

            logger.info(f"Starting training (Gen {cycle})...")
            result = wrapper.train(initial_potential=initial_potential)

            if result.success and result.potential_path:
                logger.info(f"Training complete. Potential at: {result.potential_path}")
                # Save potential to generation specific path
                pot_dir = self.manager.work_dir / "potentials"
                pot_dir.mkdir(exist_ok=True)
                dest = pot_dir / f"generation_{cycle}.yace"

                # Update state
                self.manager.state.latest_potential_path = dest

                # Copy
                try:
                    shutil.copy2(result.potential_path, dest)
                except Exception:
                    logger.exception("Failed to save potential artifacts")
            else:
                logger.error("Training failed.")

        except Exception:
            logger.exception("Training phase failed")
