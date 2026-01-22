"""
Module for training orchestration.
"""

import logging
from pathlib import Path

from mlip_autopipec.config.schemas.training import TrainingConfig, TrainingResult
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.training.dataset import DatasetBuilder
from mlip_autopipec.training.pacemaker import PacemakerWrapper

logger = logging.getLogger(__name__)


class TrainingManager:
    """
    Orchestrates the MLIP training workflow.

    This class acts as a coordinator, delegating specific tasks to specialized
    components. It ensures separation of concerns by:
    1. Delegating data extraction and file preparation to `DatasetBuilder`.
    2. Delegating external tool execution (Pacemaker) to `PacemakerWrapper`.

    This design keeps the core application logic clean and allows individual
    components (Data Prep vs Execution) to be tested or swapped independently.

    Attributes:
        db_manager: Interface to the ASE database for data retrieval.
        config: Training configuration parameters.
        work_dir: Directory where training artifacts and logs will be stored.
    """

    def __init__(self, db_manager: DatabaseManager, config: TrainingConfig, work_dir: Path) -> None:
        """
        Initialize the TrainingManager.

        Args:
            db_manager: Initialized DatabaseManager instance.
            config: Validated TrainingConfig object.
            work_dir: Path to the working directory.
        """
        self.db_manager = db_manager
        self.config = config
        self.work_dir = work_dir

    def run_training(self) -> TrainingResult:
        """
        Executes the full training pipeline.

        Steps:
        1. Export completed calculations from the database to training/test files.
        2. Configure the Pacemaker training engine.
        3. Execute the training process and parse results.

        Returns:
            TrainingResult: Object containing success status, metrics, and path
                            to the generated potential file.
        """
        try:
            logger.info("Starting training workflow...")

            # 1. Export Data
            builder = DatasetBuilder(self.db_manager)
            builder.export(self.config, self.work_dir)

            # 2. Run Pacemaker
            wrapper = PacemakerWrapper(self.config, self.work_dir)
            return wrapper.train()

        except Exception:
            logger.exception("Training workflow failed")
            return TrainingResult(success=False)
