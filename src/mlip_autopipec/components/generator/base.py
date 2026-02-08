from abc import abstractmethod
from collections.abc import Iterator
from typing import Any, TypeVar

from mlip_autopipec.domain_models.config import ComponentConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.interfaces.base_component import BaseComponent

# Define generic type constrained to ComponentConfig
ConfigT = TypeVar("ConfigT", bound=ComponentConfig)


class BaseGenerator(BaseComponent[ConfigT]):
    @property
    def name(self) -> str:
        return "generator"

    @abstractmethod
    def generate(
        self, n_structures: int, config: dict[str, Any] | None = None
    ) -> Iterator[Structure]:
        """Generate structures."""
        ...
