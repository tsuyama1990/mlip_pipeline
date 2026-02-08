from abc import abstractmethod
from collections.abc import Iterable, Iterator

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.interfaces.base_component import BaseComponent


class BaseOracle(BaseComponent):
    @property
    def name(self) -> str:
        return "oracle"

    @abstractmethod
    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """Compute labels for structures."""
        ...
