from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
from mlip_autopipec.domain_models import Dataset, StructureMetadata

class BaseExplorer(ABC):
    @abstractmethod
    def explore(self, potential_path: Optional[Path], dataset: Dataset) -> List[StructureMetadata]:
        pass
