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
    Orchestrates the training workflow:
    1. Prepare Data (Export from DB)
    2. Configure Pacemaker
    3. Run Training
    """

    def __init__(self, db_manager: DatabaseManager, config: TrainingConfig, work_dir: Path) -> None:
        self.db_manager = db_manager
        self.config = config
        self.work_dir = work_dir

    def run_training(self) -> TrainingResult:
        """
        Executes the full training pipeline.
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
