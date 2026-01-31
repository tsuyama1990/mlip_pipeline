import logging
from pathlib import Path
from typing import Optional

from mlip_autopipec.domain_models.workflow import WorkflowState

logger = logging.getLogger("mlip_autopipec.orchestration")


class StateManager:
    """
    Manages persistence of the WorkflowState.
    Uses atomic writes to prevent corruption.
    """

    def __init__(self, work_dir: Path, filename: str = "workflow_state.json"):
        self.work_dir = work_dir
        self.state_path = work_dir / filename
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: WorkflowState) -> None:
        """
        Save state to disk atomically.
        """
        tmp_path = self.state_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                f.write(state.model_dump_json(indent=2))
            tmp_path.rename(self.state_path)
            logger.debug(f"State saved to {self.state_path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def load(self) -> Optional[WorkflowState]:
        """
        Load state from disk. Returns None if file doesn't exist.
        """
        if not self.state_path.exists():
            return None

        try:
            with open(self.state_path, "r") as f:
                data = f.read()
            return WorkflowState.model_validate_json(data)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            raise
