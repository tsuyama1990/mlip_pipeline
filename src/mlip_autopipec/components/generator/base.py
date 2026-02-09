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
        self, n_structures: int, cycle: int = 0, metrics: dict[str, Any] | None = None
    ) -> Iterator[Structure]:
        """
        Generate structures.

        Args:
            n_structures: The number of structures to generate.
            cycle: The current active learning cycle number.
            metrics: Optional metrics from the previous cycle.
        """
        ...

    def enhance(self, structure: Structure) -> Iterator[Structure]:
        """
        Enhance a structure by generating local candidates (e.g. for halted structures).

        Args:
            structure: The seed structure.

        Returns:
            Iterator of enhanced structures (including the seed).
        """
        # Default implementation returns just the structure itself if not overridden
        yield structure

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
