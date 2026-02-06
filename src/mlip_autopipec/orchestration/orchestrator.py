import logging
from pathlib import Path

from ase.io import iread, write

from mlip_autopipec.config import GlobalConfig
from mlip_autopipec.domain_models import Dataset
from mlip_autopipec.interfaces import BaseExplorer, BaseOracle, BaseTrainer, BaseValidator

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        config: GlobalConfig,
        explorer: BaseExplorer,
        oracle: BaseOracle,
        trainer: BaseTrainer,
        validator: BaseValidator,
    ) -> None:
        self.config = config
        self.explorer = explorer
        self.oracle = oracle
        self.trainer = trainer
        self.validator = validator

        # Ensure work directory exists
        self.config.work_dir.mkdir(parents=True, exist_ok=True)

        # Initialize accumulated dataset file
        self.dataset_file = self.config.work_dir / "accumulated_dataset.xyz"

        # Ensure the accumulated dataset file exists, even if empty
        if not self.dataset_file.exists():
            self.dataset_file.touch()

        # Current potential path (start with initial from config or default)
        self.current_potential_path = self.config.initial_potential or Path(
            "initial_potential.yace"
        )

    def run(self) -> None:
        logger.info("Orchestrator initialization complete")

        for cycle in range(1, self.config.max_cycles + 1):
            logger.info(f"Starting Cycle {cycle}...")

            # 1. Explore
            logger.info("Running Explorer...")
            # For Cycle 01, we just pass the current full dataset.
            full_dataset = Dataset(file_path=self.dataset_file)
            new_candidates = self.explorer.explore(self.current_potential_path, full_dataset)

            logger.info(f"Explorer produced candidates at {new_candidates.file_path}.")

            # 2. Oracle
            logger.info("Running Oracle...")
            labeled_data = self.oracle.label(new_candidates)
            logger.info(f"Oracle labeled structures at {labeled_data.file_path}.")

            # 3. Accumulate (Stream to disk to avoid memory explosion)
            if labeled_data.file_path.exists():
                # Read labeled data and append to accumulated dataset
                logger.info(
                    f"Appending structures from {labeled_data.file_path} to {self.dataset_file}"
                )

                try:
                    count = 0
                    # Using ASE iread to stream structures
                    # We write to the accumulated file in append mode
                    # Using a context manager for writing might be cleaner but 'append=True' in write handles it

                    for atoms in iread(labeled_data.file_path):
                        write(self.dataset_file, atoms, format="extxyz", append=True)
                        count += 1

                    if count > 0:
                        logger.info(f"Appended {count} structures.")
                    else:
                        logger.warning("No structures found in labeled data.")
                except Exception as e:
                    msg = f"Failed to append labeled structures to dataset file: {e}"
                    logger.exception(msg)
                    raise RuntimeError(msg) from e

            # 4. Train
            logger.info("Running Trainer...")
            # Create a Dataset pointing to the accumulated file
            full_dataset = Dataset(file_path=self.dataset_file)

            # Train on full dataset (validation dataset is same for now/MVP)
            potential_path = self.trainer.train(full_dataset, full_dataset)
            self.current_potential_path = potential_path
            logger.info(f"Trainer produced potential version {cycle} at {potential_path}.")

            # 5. Validate
            validation_result = self.validator.validate(potential_path)
            logger.info(f"Validation Result: {validation_result}")

            logger.info(f"Cycle {cycle} completed.")

        logger.info("All cycles finished.")
