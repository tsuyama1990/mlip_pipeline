from abc import ABC, abstractmethod

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowState


class Phase(ABC):
    """
    Abstract base class for a workflow phase.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the phase."""
        ...

    @abstractmethod
    def execute(self, state: WorkflowState, config: Config) -> WorkflowState:
        """
        Execute the phase logic.

        Args:
            state: Current workflow state.
            config: Global configuration.

        Returns:
            Updated workflow state.
        """
        ...
