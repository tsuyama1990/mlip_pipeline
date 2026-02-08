from abc import abstractmethod
from collections.abc import Iterable, Iterator

from mlip_autopipec.domain_models.config import OracleConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.interfaces.base_component import BaseComponent


class BaseOracle(BaseComponent[OracleConfig]):
    @property
    def name(self) -> str:
        return "oracle"

    @abstractmethod
    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """
        Compute labels for structures.

        Args:
            structures: Input structures without labels.

        Returns:
            Iterator of labeled structures.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
