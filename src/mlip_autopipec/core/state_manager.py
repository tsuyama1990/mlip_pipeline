import json
import logging
from pathlib import Path

from mlip_autopipec.domain_models.datastructures import WorkflowState

logger = logging.getLogger(__name__)


class StateManager:
    """Manages the persistence of the workflow state."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.state_file = work_dir / "workflow_state.json"
        self.state = self._load_or_create()

    def _load_or_create(self) -> WorkflowState:
        """Loads state from file or creates a new one."""
        if self.state_file.exists():
            try:
                with self.state_file.open("r") as f:
                    data = json.load(f)
                return WorkflowState.model_validate(data)
            except Exception:
                logger.exception("Failed to load state file. Creating new state.")
                return WorkflowState()
        return WorkflowState()

    def save(self) -> None:
        """Atomically saves the current state to JSON."""
        tmp_file = self.state_file.with_suffix(".tmp")
        try:
            with tmp_file.open("w") as f:
                f.write(self.state.model_dump_json(indent=2))
            tmp_file.replace(self.state_file)
            logger.debug(f"State saved to {self.state_file}")
        except Exception:
            logger.exception("Failed to save state")
            if tmp_file.exists():
                tmp_file.unlink()

    def update_cycle(self, cycle: int) -> None:
        self.state.current_cycle = cycle
        self.save()

    def update_potential(self, path: Path) -> None:
        self.state.active_potential_path = path
        self.save()

    def cleanup(self) -> None:
        """Removes temporary files."""
        # For now, just remove tmp state file if exists
        tmp_file = self.state_file.with_suffix(".tmp")
        if tmp_file.exists():
            tmp_file.unlink()
        logger.info("Cleanup completed.")
