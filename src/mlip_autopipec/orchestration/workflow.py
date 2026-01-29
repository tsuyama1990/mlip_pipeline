import json
import logging
from pathlib import Path

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState
from mlip_autopipec.orchestration.protocols import Phase

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    Orchestrates the Active Learning Cycle by managing state and executing phases.
    """

    def __init__(self, config: Config, state_path: Path = Path("workflow_state.json")) -> None:
        self.config = config
        self.state_path = state_path
        self.state = self._load_state()
        self.phases: dict[WorkflowPhase, Phase] = {}

    def _load_state(self) -> WorkflowState:
        """Load state from disk or create new."""
        if self.state_path.exists():
            try:
                with self.state_path.open("r") as f:
                    data = json.load(f)
                logger.info(f"Loaded workflow state from {self.state_path}")
                return WorkflowState(**data)
            except Exception as e:
                logger.warning(f"Failed to load state from {self.state_path}: {e}. Starting fresh.")
                return WorkflowState()
        else:
            logger.info("No existing state found. Starting fresh.")
            return WorkflowState()

    def save_state(self) -> None:
        """Save current state to disk."""
        with self.state_path.open("w") as f:
            f.write(self.state.model_dump_json(indent=2))
        logger.debug(f"Saved state to {self.state_path}")

    def register_phase(self, phase: Phase, phase_enum: WorkflowPhase) -> None:
        """Register a phase handler."""
        self.phases[phase_enum] = phase

    def run_cycle(self) -> None:
        """
        Execute the current phase and handle transitions.
        """
        if self.state.cycle_index >= self.config.orchestrator.max_cycles:
            logger.info("Maximum cycles reached. Workflow completed.")
            return

        current_phase_enum = self.state.current_phase
        phase_handler = self.phases.get(current_phase_enum)

        if not phase_handler:
            msg = f"No handler registered for phase {current_phase_enum}"
            logger.error(msg)
            raise ValueError(msg)

        logger.info(f"--- Starting Phase: {current_phase_enum.value} (Cycle {self.state.cycle_index}) ---")

        try:
            # Execute the phase
            new_state = phase_handler.execute(self.state, self.config)
            self.state = new_state
            self.save_state()

            # Handle Transition
            self._handle_transition()
            self.save_state()

        except Exception:
            logger.exception(f"Error during phase {current_phase_enum}")
            raise

    def _handle_transition(self) -> None:
        """
        Determine the next phase based on the current state.
        """
        current = self.state.current_phase

        # Define standard loop
        order = [
            WorkflowPhase.EXPLORATION,
            WorkflowPhase.SELECTION,
            WorkflowPhase.CALCULATION,
            WorkflowPhase.TRAINING,
            WorkflowPhase.VALIDATION
        ]

        if self.state.is_halted and current == WorkflowPhase.EXPLORATION:
            # Halt detected -> Go to Selection (or specific handling)
            logger.info("Halt detected in Exploration. Proceeding to Selection.")
            self.state.current_phase = WorkflowPhase.SELECTION
            return

        # Default transition
        try:
            idx = order.index(current)
            if idx == len(order) - 1:
                # End of cycle
                self.state.cycle_index += 1
                self.state.current_phase = WorkflowPhase.EXPLORATION
                logger.info(f"Cycle completed. Advancing to Cycle {self.state.cycle_index}")
            else:
                next_phase = order[idx + 1]
                self.state.current_phase = next_phase
                logger.info(f"Transitioning to {next_phase.value}")
        except ValueError:
            logger.warning(f"Current phase {current} not in standard order. No transition applied.")
