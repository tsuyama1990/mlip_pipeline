from abc import ABC, abstractmethod
from typing import List
from mlip_autopipec.domain_models import StructureMetadata

class BaseOracle(ABC):
    @abstractmethod
    def label(self, structures: List[StructureMetadata]) -> List[StructureMetadata]:
        pass
