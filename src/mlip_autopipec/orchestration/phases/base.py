from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlip_autopipec.orchestration.workflow import WorkflowManager

class BasePhase(ABC):
    """
    Abstract base class for workflow phases.
    """
    def __init__(self, manager: "WorkflowManager") -> None:
        self.manager = manager
        self.config = manager.config
        self.db = manager.db_manager
        self.queue = manager.task_queue

    @abstractmethod
    def execute(self) -> Any:
        """Execute the phase logic."""
