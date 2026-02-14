"""Base module interface."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.logging import get_logger


class ModuleResult(BaseModel):
    """Standardized result from a module execution.

    Attributes:
        status: The final status of the execution (e.g., 'success', 'failed').
        metrics: A dictionary of key performance indicators or results (e.g., 'energy', 'forces').
        artifacts: A dictionary mapping artifact names to their file paths on disk.

    """

    model_config = ConfigDict(extra="forbid")
    status: str = Field(..., description="Execution status (success, failed)")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Execution metrics")
    artifacts: dict[str, str] = Field(
        default_factory=dict, description="Paths to generated artifacts"
    )


class BaseModule(ABC):
    """Abstract Base Class for all PYACEMAKER modules.

    All core components (Oracle, Trainer, etc.) must inherit from this class.
    It provides a standardized initialization with configuration injection
    and automatic logger setup.
    """

    def __init__(self, config: PYACEMAKERConfig) -> None:
        """Initialize the module.

        Args:
            config: The validated PYACEMAKER configuration.

        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def run(self) -> ModuleResult:
        """Execute the module's main logic.

        This method should encapsulate the module's primary workflow.
        It is expected to handle internal errors gracefully where possible,
        or raise specialized exceptions defined in `pyacemaker.core.exceptions`
        if the failure is unrecoverable.

        Returns:
            ModuleResult containing status, metrics, and artifact paths.

        Raises:
            PYACEMAKERError: If a critical error occurs during execution.

        """
        ...
