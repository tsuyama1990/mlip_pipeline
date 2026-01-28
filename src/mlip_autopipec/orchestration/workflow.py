import logging
from pathlib import Path

from mlip_autopipec.config.schemas.core import UserInputConfig
from mlip_autopipec.data_models.state import WorkflowState
from mlip_autopipec.orchestration.database import DatabaseManager

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    Orchestrates the active learning cycles.
    """

    def __init__(self, config: UserInputConfig, work_dir: Path, state_file: Path | None = None):
        self.config = config
        self.work_dir = work_dir
        self.state_file = state_file or (work_dir / "state.json")
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Initialize DB
        # Use config value for db path, not hardcoded
        self.db_path = self.work_dir / self.config.runtime.database_path
        self.db_manager = DatabaseManager(self.db_path)

        # Initialize State
        self.state = self._load_state()

    def _load_state(self) -> WorkflowState:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return WorkflowState.model_validate_json(f.read())
        return WorkflowState()

    def save_state(self) -> None:
        with open(self.state_file, "w") as f:
            f.write(self.state.model_dump_json(indent=2))

    def run(self) -> None:
        """Main execution loop."""
        max_cycles = self.config.workflow.max_generations

        while self.state.cycle_index < max_cycles:
            logger.info(f"Starting Cycle {self.state.cycle_index}")

            # 1. Generation Phase
            self.run_generation()

            # 2. DFT Phase
            self.run_dft()

            # 3. Training Phase
            self.run_training()

            self.state.cycle_index += 1
            self.save_state()

    def run_generation(self):
        # Implementation placeholder
        pass

    def run_dft(self):
        # Implementation placeholder
        pass

    def run_training(self):
        # Implementation placeholder
        pass
