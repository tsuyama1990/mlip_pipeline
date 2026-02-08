from abc import ABC, abstractmethod
from typing import Any


class BaseComponent(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the component."""
        ...
