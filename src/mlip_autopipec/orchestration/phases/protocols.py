from typing import Protocol, List, Optional, Callable
from pathlib import Path

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import CandidateStructure, WorkflowState
from mlip_autopipec.domain_models.dynamics import LammpsResult

class IExplorationPhase(Protocol):
    def execute(self, state: WorkflowState, config: Config, work_dir: Path) -> LammpsResult:
        """
        Run structure generation and MD exploration.
        Returns the result of the MD simulation.
        """
        ...

class ISelectionPhase(Protocol):
    def execute(self, state: WorkflowState, config: Config, work_dir: Path) -> List[CandidateStructure]:
        """
        Select candidates from the exploration trajectory.
        Returns a list of selected candidates.
        """
        ...

class ICalculationPhase(Protocol):
    def execute(self, state: WorkflowState, config: Config, work_dir: Path, save_state_callback: Optional[Callable[[], None]] = None) -> bool:
        """
        Run DFT calculations for pending candidates and update the dataset.
        Returns True if successful (some candidates calculated), False otherwise.
        Optional save_state_callback can be used to persist state during long-running batch processing.
        """
        ...

class ITrainingPhase(Protocol):
    def execute(self, state: WorkflowState, config: Config, work_dir: Path) -> Optional[Path]:
        """
        Train the potential using the accumulated dataset.
        Returns the path to the new potential if successful, None otherwise.
        """
        ...

class IValidationPhase(Protocol):
    def execute(self, state: WorkflowState, config: Config, work_dir: Path) -> bool:
        """
        Validate the current potential.
        Returns True if validation ran (status is logged/reported), False if failed to run.
        """
        ...
