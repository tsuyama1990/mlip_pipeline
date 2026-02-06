import logging

from ase.io import iread, read, write

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
        self.dataset_file = self.config.work_dir / "accumulated_dataset.xyz"
        # If file doesn't exist, create it (empty) or just let append handle it?
        # ASE write append=True works if file doesn't exist, but creates it.

        # We need a placeholder input dataset for Explorer (usually requires potential + data).
        # We'll create an empty placeholder file for the first cycle.
        self.empty_input_file = self.config.work_dir / "empty_input.xyz"
        if not self.empty_input_file.exists():
             self.empty_input_file.touch()

        # Current potential path (start with initial from config or default relative to work_dir)
        self.current_potential_path = self.config.initial_potential or (self.config.work_dir / "initial_potential.yace")

        # Track total structures if limit is set
        self.total_structures = 0
        if self.dataset_file.exists():
            # Count existing structures (expensive but necessary for restart safety?)
            # For now assume 0 or count them.
            # Minimal approach: count on init.
            try:
                self.total_structures = len(read(self.dataset_file, index=":"))
            except Exception:
                self.total_structures = 0

    def run(self) -> None:
        logger.info("Orchestrator initialization complete")

        for cycle in range(1, self.config.max_cycles + 1):
            logger.info(f"Starting Cycle {cycle}...")

            # 1. Explore
            logger.info("Running Explorer...")
            new_candidates = self.explorer.explore(
                self.current_potential_path,
                Dataset(file_path=self.empty_input_file)
            )
            logger.info(f"Explorer produced candidates at {new_candidates.file_path}.")

            # 2. Oracle
            logger.info("Running Oracle...")
            labeled_data = self.oracle.label(new_candidates)

            # 3. Accumulate
            logger.info(f"Appending structures from {labeled_data.file_path} to {self.dataset_file}")

            # Check limits
            limit_reached = False
            if (
                self.config.max_accumulated_structures is not None
                and self.total_structures >= self.config.max_accumulated_structures
            ):
                logger.warning("Max accumulated structures limit reached. Stopping accumulation.")
                limit_reached = True

            if not limit_reached:
                count = 0
                for atoms in iread(labeled_data.file_path):
                    if (
                        self.config.max_accumulated_structures is not None
                        and self.total_structures >= self.config.max_accumulated_structures
                    ):
                        logger.warning("Max accumulated structures limit reached during appending.")
                        break

                    write(self.dataset_file, atoms, append=True)
                    self.total_structures += 1
                    count += 1
                logger.info(f"Appended {count} structures. Total: {self.total_structures}")

            # 4. Train
            logger.info("Running Trainer...")
            # Create a Dataset pointing to the file
            full_dataset = Dataset(file_path=self.dataset_file)

            potential_path = self.trainer.train(full_dataset, full_dataset)
            self.current_potential_path = potential_path
            logger.info(f"Trainer produced potential version {cycle} at {potential_path}.")

            # 5. Validate
            validation_result = self.validator.validate(potential_path)
            logger.info(f"Validation Result: {validation_result}")

            logger.info(f"Cycle {cycle} completed.")

        logger.info("All cycles finished.")
