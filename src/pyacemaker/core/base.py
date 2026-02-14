"""Base module interface."""

from abc import ABC, abstractmethod
from typing import Any

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.logging import get_logger


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
    def run(self) -> dict[str, Any]:
        """Execute the module's main logic.

        This method should encapsulate the module's primary workflow.
        It is expected to handle internal errors gracefully where possible,
        or raise specialized exceptions defined in `pyacemaker.core.exceptions`
        if the failure is unrecoverable.

        Returns:
            A dictionary containing the execution results (metrics, status, paths).

        Raises:
            PYACEMAKERError: If a critical error occurs during execution.

        """
        ...
