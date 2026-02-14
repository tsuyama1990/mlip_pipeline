"""Base module interface."""

from abc import ABC, abstractmethod
from typing import Any

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.logging import get_logger


class BaseModule(ABC):
    """Abstract Base Class for all PYACEMAKER modules."""

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

        Returns:
            A dictionary containing the execution results.

        """
        ...
