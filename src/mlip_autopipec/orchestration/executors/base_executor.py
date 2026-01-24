"""
Abstract base class for Phase Executors.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlip_autopipec.orchestration.workflow import WorkflowManager

class BaseExecutor(ABC):
    """
    Abstract base class for executing specific phases of the workflow.
    """

    def __init__(self, manager: "WorkflowManager") -> None:
        self.manager = manager
        self.config = manager.config
        self.db = manager.db_manager
        self.queue = manager.task_queue

    @abstractmethod
    def execute(self) -> bool:
        """
        Execute the phase logic.

        Returns:
            bool: True if the phase produced actionable results (e.g. new candidates), False otherwise.
        """
        pass
