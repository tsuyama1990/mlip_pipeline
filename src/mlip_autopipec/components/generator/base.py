from abc import abstractmethod
from collections.abc import Iterator
from typing import Any

from mlip_autopipec.domain_models.config import GeneratorConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.interfaces.base_component import BaseComponent


class BaseGenerator(BaseComponent[GeneratorConfig]):
    @property
    def name(self) -> str:
        return "generator"

    @abstractmethod
    def generate(
        self, n_structures: int, config: dict[str, Any] | None = None
    ) -> Iterator[Structure]:
        """Generate structures."""
        ...
