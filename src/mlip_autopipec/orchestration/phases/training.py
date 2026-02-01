import logging
from pathlib import Path
from typing import Optional

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner

logger = logging.getLogger("mlip_autopipec.phases.training")

class TrainingPhase:
    def execute(self, state: WorkflowState, config: Config, work_dir: Path) -> Optional[Path]:
        logger.info("Training potential...")

        if not config.training:
             raise ValueError("Training configuration missing.")

        train_dir = work_dir / "training"
        pacemaker = PacemakerRunner(
            work_dir=train_dir,
            train_config=config.training,
            potential_config=config.potential
        )

        # Set initial potential
        if state.latest_potential_path:
            pacemaker.train_config.initial_potential = state.latest_potential_path

        result = pacemaker.train(state.dataset_path)

        if result.status != JobStatus.COMPLETED:
            raise RuntimeError(f"Training failed: {result.log_content}")

        return result.potential_path
