import logging
from pathlib import Path

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState
from mlip_autopipec.orchestration.state import StateManager
from mlip_autopipec.orchestration.phases.exploration import ExplorationPhase
from mlip_autopipec.orchestration.phases.selection import SelectionPhase
from mlip_autopipec.orchestration.phases.calculation import CalculationPhase
from mlip_autopipec.orchestration.phases.training import TrainingPhase
from mlip_autopipec.orchestration.phases.validation import ValidationPhase

logger = logging.getLogger("mlip_autopipec.orchestration")

class Orchestrator:
    """
    The Brain of the Active Learning Loop.
    Manages state transitions and delegates to Phase workers.
    """

    def __init__(self, config: Config, work_dir: Path = Path(".")):
        self.config = config
        self.work_dir = work_dir
        self.state_manager = StateManager(work_dir)

        # Phases
        self.exploration_phase = ExplorationPhase()
        self.selection_phase = SelectionPhase()
        self.calculation_phase = CalculationPhase()
        self.training_phase = TrainingPhase()
        self.validation_phase = ValidationPhase()

        # Load or Init State
        loaded_state = self.state_manager.load()
        if loaded_state:
            self.state: WorkflowState = loaded_state
            logger.info(f"Resumed state: Gen {self.state.generation}, Phase {self.state.current_phase}")
        else:
            logger.info("No existing state found. Initializing new WorkflowState.")
            self.state = WorkflowState(
                project_name=config.project_name,
                dataset_path=work_dir / "data" / "accumulated.pckl.gzip",
                current_phase=WorkflowPhase.EXPLORATION,
                latest_potential_path=config.training.initial_potential if config.training else None
            )
            self.state_manager.save(self.state)

        # Ensure data dir exists (phases rely on it)
        self.state.dataset_path.parent.mkdir(parents=True, exist_ok=True)

    def run_loop(self):
        """
        Main execution loop.
        """
        logger.info("Starting Autonomous Active Learning Loop")

        while self.state.generation < self.config.orchestrator.max_iterations:
            should_continue = self.step()
            if not should_continue:
                break

        logger.info("Loop finished.")

    def step(self) -> bool:
        """
        Executes one step of the workflow based on the current phase.
        Returns True if the loop should continue, False if it should stop.
        """
        phase = self.state.current_phase
        gen_dir = self.work_dir / f"active_learning/iter_{self.state.generation:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        try:
            if phase == WorkflowPhase.EXPLORATION:
                result = self.exploration_phase.execute(self.state, self.config, gen_dir)

                # Detection Logic (Orchestrator decides)
                halt_detected = False
                if result.max_gamma is not None:
                     if result.max_gamma > self.config.orchestrator.uncertainty_threshold:
                         logger.info(f"Halt detected! Max Gamma: {result.max_gamma}")
                         halt_detected = True

                if halt_detected:
                    self.transition_to(WorkflowPhase.SELECTION)
                else:
                    logger.info("Exploration finished without halt. Convergence detected?")
                    return False

            elif phase == WorkflowPhase.SELECTION:
                candidates = self.selection_phase.execute(self.state, self.config, gen_dir)
                self.state.candidates = candidates
                self.transition_to(WorkflowPhase.CALCULATION)

            elif phase == WorkflowPhase.CALCULATION:
                # Pass save callback
                def save_callback() -> None:
                    self.state_manager.save(self.state)

                success = self.calculation_phase.execute(self.state, self.config, gen_dir, save_state_callback=save_callback)
                if success:
                    self.transition_to(WorkflowPhase.TRAINING)
                else:
                    logger.error("Calculation phase failed (no valid candidates processed).")
                    return False

            elif phase == WorkflowPhase.TRAINING:
                new_pot = self.training_phase.execute(self.state, self.config, gen_dir)
                if new_pot is not None:
                    self.state.latest_potential_path = new_pot
                    self.transition_to(WorkflowPhase.VALIDATION)
                else:
                    logger.error("Training returned no potential path.")
                    return False

            elif phase == WorkflowPhase.VALIDATION:
                self.validation_phase.execute(self.state, self.config, gen_dir)
                self.state.generation += 1
                self.transition_to(WorkflowPhase.EXPLORATION)

            return True

        except Exception as e:
            logger.exception(f"Error in phase {phase}: {e}")
            return False

    def transition_to(self, new_phase: WorkflowPhase):
        logger.info(f"Transition: {self.state.current_phase} -> {new_phase}")
        self.state.current_phase = new_phase
        self.state_manager.save(self.state)
