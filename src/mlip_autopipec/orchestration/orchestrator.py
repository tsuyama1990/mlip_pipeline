import logging
from pathlib import Path

from ase.io import write

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
        validator: BaseValidator
    ) -> None:
        self.config = config
        self.explorer = explorer
        self.oracle = oracle
        self.trainer = trainer
        self.validator = validator

        # Ensure work directory exists
        self.config.work_dir.mkdir(parents=True, exist_ok=True)

        # Initialize accumulated dataset file
        self.dataset_file = self.config.work_dir / self.config.dataset_file_name

        # Current potential path (start with initial from config or default)
        self.current_potential_path = self.config.initial_potential or Path("initial_potential.yace")

    def run(self) -> None:
        logger.info("Orchestrator initialization complete")

        for cycle in range(1, self.config.max_cycles + 1):
            logger.info(f"Starting Cycle {cycle}...")

            # 1. Explore
            logger.info("Running Explorer...")
            # We pass an empty dataset or partial dataset to explore?
            # Usually explore needs the potential.
            # For Cycle 01, we just pass an empty one as placeholder.
            new_candidates = self.explorer.explore(self.current_potential_path, Dataset(structures=[]))
            logger.info(f"Explorer produced {len(new_candidates.structures)} candidates.")

            # 2. Oracle
            logger.info("Running Oracle...")
            labeled_data = self.oracle.label(new_candidates)
            logger.info(f"Oracle labeled {len(labeled_data.structures)} structures.")

            # 3. Accumulate (Stream to disk to avoid memory explosion)
            if labeled_data.structures:
                # Append to file
                logger.info(f"Appending {len(labeled_data.structures)} structures to {self.dataset_file}")
                # Extract ASE atoms
                atoms_list = [s.structure for s in labeled_data.structures]
                write(self.dataset_file, atoms_list, append=True)

            # 4. Train
            logger.info("Running Trainer...")
            # Create a Dataset pointing to the file, empty structures list to save memory
            full_dataset = Dataset(file_path=self.dataset_file, structures=[])

            potential_path = self.trainer.train(full_dataset, full_dataset)
            self.current_potential_path = potential_path
            logger.info(f"Trainer produced potential version {cycle} at {potential_path}.")

            # 5. Validate
            validation_result = self.validator.validate(potential_path)
            logger.info(f"Validation Result: {validation_result}")

            logger.info(f"Cycle {cycle} completed.")

        logger.info("All cycles finished.")
