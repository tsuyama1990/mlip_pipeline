from abc import ABC, abstractmethod


class BaseOrchestrator(ABC):
    """
    Abstract base class for the Orchestrator.
    """

    @abstractmethod
    def run(self) -> None:
        """
        Executes the active learning loop.
        """
