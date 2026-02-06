from abc import ABC, abstractmethod
from pathlib import Path

from mlip_autopipec.domain_models import Dataset, StructureMetadata


class BaseExplorer(ABC):
    @abstractmethod
    def explore(self, potential_path: Path | None, dataset: Dataset) -> list[StructureMetadata]:
        pass
