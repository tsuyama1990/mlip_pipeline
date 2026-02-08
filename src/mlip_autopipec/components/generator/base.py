from abc import abstractmethod
from collections.abc import Iterator

from mlip_autopipec.domain_models.config import GeneratorConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.interfaces.base_component import BaseComponent


class BaseGenerator(BaseComponent[GeneratorConfig]):
    @property
    def name(self) -> str:
        return "generator"

    @abstractmethod
    def generate(self, n_structures: int) -> Iterator[Structure]:
        """
        Generate structures.

        Args:
            n_structures: The number of structures to generate.
        """
        ...
