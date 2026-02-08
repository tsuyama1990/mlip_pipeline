from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from mlip_autopipec.domain_models.config import ComponentConfig

ConfigT = TypeVar("ConfigT", bound=ComponentConfig)


class BaseComponent(ABC, Generic[ConfigT]):
    def __init__(self, config: ConfigT) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the component."""
        ...
