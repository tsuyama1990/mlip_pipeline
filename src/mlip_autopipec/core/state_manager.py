import contextlib
import json
import logging
import tempfile
from pathlib import Path

from mlip_autopipec.domain_models.enums import TaskType
from mlip_autopipec.domain_models.workflow import WorkflowState

logger = logging.getLogger(__name__)


class StateManager:
    """Manages the persistence of the workflow state."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.state_file = work_dir / "workflow_state.json"
        self.state = self._load_or_create()
        self.dirty = False  # Track if state has changed

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

    def save(self, force: bool = False) -> None:
        """Atomically saves the current state to JSON if dirty or forced."""
        if not self.dirty and not force:
            return

        tmp_path = None
        try:
            # Use tempfile in the same directory to ensure atomic move support
            # Use delete=False so we can rename it later
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.work_dir, delete=False, suffix=".tmp"
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                tmp_file.write(self.state.model_dump_json(indent=2))

            # File is closed now. Safe to rename on all platforms.
            tmp_path.replace(self.state_file)
            logger.debug(f"State saved to {self.state_file}")
            self.dirty = False
        except Exception:
            logger.exception("Failed to save state")
            # Attempt cleanup if tmp file exists and wasn't moved
            if tmp_path and tmp_path.exists():
                with contextlib.suppress(OSError):
                    tmp_path.unlink()

    def update_cycle(self, cycle: int) -> None:
        if self.state.current_cycle != cycle:
            self.state.current_cycle = cycle
            self.dirty = True

    def update_step(self, step: TaskType) -> None:
        if self.state.current_step != step:
            self.state.current_step = step
            self.dirty = True

    def update_potential(self, path: Path) -> None:
        if self.state.active_potential_path != path:
            self.state.active_potential_path = path
            self.dirty = True

    def cleanup(self) -> None:
        """Removes temporary files."""
        # Clean up any .tmp files in work_dir that look like state files
        # Pattern: *.tmp
        for p in self.work_dir.glob("*.tmp"):
            with contextlib.suppress(OSError):
                p.unlink()
        logger.info("Cleanup completed.")
