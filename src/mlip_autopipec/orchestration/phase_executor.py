"""
Refactored PhaseExecutor that delegates to specialized executors.
"""

from typing import TYPE_CHECKING

from mlip_autopipec.orchestration.executors.dft_executor import DFTExecutor
from mlip_autopipec.orchestration.executors.exploration_executor import ExplorationExecutor
from mlip_autopipec.orchestration.executors.inference_executor import InferenceExecutor
from mlip_autopipec.orchestration.executors.training_executor import TrainingExecutor

if TYPE_CHECKING:
    from mlip_autopipec.orchestration.workflow import WorkflowManager

class PhaseExecutor:
    """
    Facade for executing individual workflow phases.
    Delegates actual execution to specialized classes.
    """

    def __init__(self, manager: "WorkflowManager") -> None:
        self.manager = manager
        self.exploration = ExplorationExecutor(manager)
        self.dft = DFTExecutor(manager)
        self.training = TrainingExecutor(manager)
        self.inference = InferenceExecutor(manager)

    def execute_exploration(self) -> None:
        self.exploration.execute()

    def execute_dft(self) -> None:
        self.dft.execute()

    def execute_training(self) -> None:
        self.training.execute()

    def execute_inference(self) -> bool:
        return self.inference.execute()
