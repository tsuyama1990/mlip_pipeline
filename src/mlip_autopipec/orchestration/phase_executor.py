from typing import TYPE_CHECKING

from mlip_autopipec.orchestration.phases.dft import DFTPhase
from mlip_autopipec.orchestration.phases.exploration import ExplorationPhase
from mlip_autopipec.orchestration.phases.inference import InferencePhase
from mlip_autopipec.orchestration.phases.selection import SelectionPhase
from mlip_autopipec.orchestration.phases.training import TrainingPhase

if TYPE_CHECKING:
    from mlip_autopipec.orchestration.workflow import WorkflowManager

class PhaseExecutor:
    """
    Facade for executing workflow phases.
    Delegates actual logic to dedicated phase classes.
    """

    def __init__(self, manager: "WorkflowManager") -> None:
        self.exploration_phase = ExplorationPhase(manager)
        self.dft_phase = DFTPhase(manager)
        self.training_phase = TrainingPhase(manager)
        self.inference_phase = InferencePhase(manager)
        self.selection_phase = SelectionPhase(manager)

    def execute_exploration(self) -> None:
        """Execute Phase A: Exploration."""
        self.exploration_phase.execute()

    def execute_dft(self) -> None:
        """Execute Phase B: DFT Labeling."""
        self.dft_phase.execute()

    def execute_training(self) -> None:
        """Execute Phase C: Training."""
        self.training_phase.execute()

    def execute_inference(self) -> bool:
        """
        Execute Phase: Exploration (MD Inference).
        Returns: True if high uncertainty was detected (halted), False otherwise.
        """
        return self.inference_phase.execute()

    def execute_selection(self) -> None:
        """Execute Phase: Selection."""
        self.selection_phase.execute()
