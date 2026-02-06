from abc import ABC, abstractmethod

from mlip_autopipec.domain_models import StructureMetadata


class BaseOracle(ABC):
    @abstractmethod
    def label(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        pass
