import logging
from pathlib import Path

from ase.io import iread, write

from mlip_autopipec.config import GlobalConfig
from mlip_autopipec.domain_models import Dataset
from mlip_autopipec.interfaces import BaseExplorer, BaseOracle, BaseTrainer, BaseValidator

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrates the active learning pipeline cycles.
    """

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
        try:
            self.config.work_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            msg = f"Failed to create work directory {self.config.work_dir}: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

        # Initialize accumulated dataset file
        self.dataset_file = self.config.work_dir / "accumulated_dataset.xyz"

        # Ensure the accumulated dataset file exists, even if empty
        if not self.dataset_file.exists():
            try:
                self.dataset_file.touch()
            except OSError as e:
                msg = f"Failed to create dataset file {self.dataset_file}: {e}"
                logger.exception(msg)
                raise RuntimeError(msg) from e

        # Current potential path (start with initial from config or default)
        self.current_potential_path = self.config.initial_potential or Path("initial_potential.yace")

    def run(self) -> None:
        """
        Executes the main active learning loop.
        """
        logger.info("Orchestrator initialization complete")

        for cycle in range(1, self.config.max_cycles + 1):
            logger.info(f"Starting Cycle {cycle}...")
            self._run_cycle(cycle)
            logger.info(f"Cycle {cycle} completed.")

        logger.info("All cycles finished.")

    def _run_cycle(self, cycle: int) -> None:
        """
        Executes a single active learning cycle.
        """
        # 1. Explore
        new_candidates = self._explore(cycle)

        # 2. Oracle
        labeled_data = self._label(cycle, new_candidates)

        # 3. Accumulate (Stream to disk to avoid memory explosion)
        self._accumulate(labeled_data)

        # 4. Train
        potential_path = self._train(cycle)

        # 5. Validate
        self._validate(cycle, potential_path)

    def _explore(self, cycle: int) -> Dataset:
        logger.info("Running Explorer...")
        full_dataset = Dataset(file_path=self.dataset_file)
        try:
            new_candidates = self.explorer.explore(self.current_potential_path, full_dataset)
        except Exception as e:
            msg = f"Explorer failed in cycle {cycle}: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

        logger.info(f"Explorer produced candidates at {new_candidates.file_path}.")
        return new_candidates

    def _label(self, cycle: int, candidates: Dataset) -> Dataset:
        logger.info("Running Oracle...")
        try:
            labeled_data = self.oracle.label(candidates)
        except Exception as e:
            msg = f"Oracle failed in cycle {cycle}: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

        logger.info(f"Oracle labeled structures at {labeled_data.file_path}.")
        return labeled_data

    def _accumulate(self, labeled_data: Dataset) -> None:
        if labeled_data.file_path.exists():
            logger.info(f"Appending structures from {labeled_data.file_path} to {self.dataset_file}")
            try:
                count = 0
                # Using ASE iread/write to append
                # Check if file is empty first to avoid ASE read errors on empty files
                if labeled_data.file_path.stat().st_size > 0:
                    for atoms in iread(labeled_data.file_path):
                        write(self.dataset_file, atoms, format="extxyz", append=True)
                        count += 1
                    logger.info(f"Appended {count} structures.")
                else:
                    logger.warning("Labeled data file is empty.")
            except Exception as e:
                msg = f"Failed to append labeled structures to dataset file: {e}"
                logger.exception(msg)
                raise RuntimeError(msg) from e
        else:
            msg = f"Labeled data file {labeled_data.file_path} not found."
            logger.error(msg)
            raise FileNotFoundError(msg)

    def _train(self, cycle: int) -> Path:
        logger.info("Running Trainer...")
        # Create a Dataset pointing to the accumulated file
        full_dataset = Dataset(file_path=self.dataset_file)
        try:
            # Train on full dataset (validation dataset is same for now/MVP)
            potential_path = self.trainer.train(full_dataset, full_dataset)
        except Exception as e:
            msg = f"Trainer failed in cycle {cycle}: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

        self.current_potential_path = potential_path
        logger.info(f"Trainer produced potential version {cycle} at {potential_path}.")
        return potential_path

    def _validate(self, cycle: int, potential_path: Path) -> None:
        try:
            validation_result = self.validator.validate(potential_path)
            logger.info(f"Validation Result: {validation_result}")
        except Exception as e:
             msg = f"Validator failed in cycle {cycle}: {e}"
             logger.exception(msg)
             raise RuntimeError(msg) from e
