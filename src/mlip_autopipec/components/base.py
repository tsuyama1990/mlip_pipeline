from abc import ABC, abstractmethod
from typing import Any, Dict, List, Iterator, Optional
from pathlib import Path
from mlip_autopipec.domain_models.config import BaseConfig

# Since actual structures might be ASE Atoms or custom Structure objects, we use Any for now or specific if possible.
# Memory says "Structure domain model implements a recursive `sanitize_value`...".
# For now, let's use Any for Structure to avoid complex dependencies in Cycle 01,
# or I can define a dummy Structure class if needed. But Any is safer for now.
Structure = Any
Potential = Any


class BaseComponent(ABC):
    """Abstract base class for all pipeline components."""

    def __init__(self, config: BaseConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)


class BaseGenerator(BaseComponent):
    """Abstract base class for structure generators."""

    @abstractmethod
    def generate(
        self, n_structures: int, cycle: int = 0, metrics: Optional[Dict[str, Any]] = None
    ) -> Iterator[Structure]:
        """Generates structures based on the configuration and cycle metrics."""
        pass

    @abstractmethod
    def enhance(self, structure: Structure) -> Iterator[Structure]:
        """Generates local candidates around a seed structure (e.g. from halt)."""
        pass


class BaseOracle(BaseComponent):
    """Abstract base class for oracle (DFT/Static Calc)."""

    @abstractmethod
    def compute(self, structure: Structure) -> Structure:
        """Computes properties (energy, forces, stress) for a structure."""
        pass

    @abstractmethod
    def compute_batch(self, structures: Iterator[Structure]) -> Iterator[Structure]:
        """Computes properties for a batch of structures."""
        pass


class BaseTrainer(BaseComponent):
    """Abstract base class for MLIP trainers."""

    @abstractmethod
    def train(
        self,
        dataset: List[Structure],
        initial_potential: Optional[Potential] = None
    ) -> Potential:
        """Trains a potential on the given dataset."""
        pass


class BaseDynamics(BaseComponent):
    """Abstract base class for dynamics engines (MD/kMC)."""

    @abstractmethod
    def explore(
        self,
        potential: Potential,
        initial_structure: Structure,
        cycle: int = 0,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Runs exploration using the potential."""
        pass


class BaseValidator(BaseComponent):
    """Abstract base class for validators."""

    @abstractmethod
    def validate(self, potential: Potential) -> Dict[str, Any]:
        """Validates the potential against test set and physical constraints."""
        pass
