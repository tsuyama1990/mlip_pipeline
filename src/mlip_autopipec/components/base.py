from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional
from pathlib import Path
from mlip_autopipec.domain_models.config import BaseConfig
from mlip_autopipec.domain_models.datastructures import Dataset, Structure

# Use Any for Structure and Potential for now until defined properly
Potential = Any

class BaseComponent(ABC):
    """Abstract base class for all pipeline components."""

    def __init__(self, config: BaseConfig, work_dir: Path) -> None:
        """
        Initialize the base component.

        Args:
            config: The component configuration object.
            work_dir: The directory for component execution.

        Raises:
            PermissionError: If the work directory cannot be created or accessed.
        """
        self.config = config
        self.work_dir = work_dir

        # Security/Maintainability: Validate work_dir path and permissions
        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            # Check write permission
            test_file = self.work_dir / ".permission_check"
            test_file.touch()
            test_file.unlink()
        except OSError as e:
            raise PermissionError(f"Cannot access or create work directory {work_dir}: {e}") from e

class BaseGenerator(BaseComponent):
    """Abstract base class for structure generators."""

    @abstractmethod
    def generate(
        self, n_structures: int, cycle: int = 0, metrics: Optional[Dict[str, Any]] = None
    ) -> Iterator[Structure]:
        """
        Generates structures based on the configuration and cycle metrics.
        Returns an iterator to allow streaming and avoid memory issues.
        """
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
        """
        Computes properties for a batch of structures.
        Must return an iterator to allow streaming.
        """
        pass

class BaseTrainer(BaseComponent):
    """Abstract base class for MLIP trainers."""

    @abstractmethod
    def train(
        self,
        dataset: Dataset,
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
