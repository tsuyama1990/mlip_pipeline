from abc import ABC, abstractmethod


class BaseOrchestrator(ABC):
    """
    Abstract Base Class for Orchestrator components.
    """

    @abstractmethod
    def run(self) -> None:
        """
        Execute the orchestration pipeline.
        """
