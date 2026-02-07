from abc import ABC, abstractmethod


class BaseOrchestrator(ABC):
    """
    Abstract Base Class for the Pipeline Orchestrator.
    """

    @abstractmethod
    def run(self) -> None:
        """
        Execute the pipeline loop.
        """
