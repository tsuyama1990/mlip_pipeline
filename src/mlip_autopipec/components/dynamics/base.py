from abc import abstractmethod
from collections.abc import Iterable, Iterator

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.interfaces.base_component import BaseComponent


class BaseDynamics(BaseComponent):
    @property
    def name(self) -> str:
        return "dynamics"

    @abstractmethod
    def explore(
        self, potential: Potential, start_structures: Iterable[Structure]
    ) -> Iterator[Structure]:
        """Explore and find uncertain structures."""
        ...
