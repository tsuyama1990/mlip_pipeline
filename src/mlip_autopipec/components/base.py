from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional
from pathlib import Path
from mlip_autopipec.domain_models.config import BaseConfig
from mlip_autopipec.domain_models.datastructures import Dataset, Structure

# Use Any for Potential for now until defined properly
Potential = Any

class BaseComponent(ABC):
    """
    Abstract base class for all pipeline components.

    Attributes:
        config (BaseConfig): The configuration object for the component.
        work_dir (Path): The working directory for the component's output.
    """

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
    """
    Abstract base class for structure generators.
    Responsible for exploring chemical/configurational space.
    """

    @abstractmethod
    def generate(
        self, n_structures: int, cycle: int = 0, metrics: Optional[Dict[str, Any]] = None
    ) -> Iterator[Structure]:
        """
        Generates structures based on the configuration and cycle metrics.
        Returns an iterator to allow streaming and avoid memory issues.

        Args:
            n_structures: Number of structures to generate.
            cycle: Current active learning cycle number.
            metrics: Optional metrics from previous cycles to guide generation.

        Returns:
            Iterator[Structure]: A stream of generated structures.
        """
        pass

    @abstractmethod
    def enhance(self, structure: Structure) -> Iterator[Structure]:
        """
        Generates local candidates around a seed structure (e.g. from halt).
        Used for local exploration in response to high uncertainty.

        Args:
            structure: The seed structure to enhance.

        Returns:
            Iterator[Structure]: A stream of enhanced candidate structures.
        """
        pass

class BaseOracle(BaseComponent):
    """
    Abstract base class for oracles (DFT/Static Calculators).
    Responsible for computing ground truth labels (energy, forces, stress).
    """

    @abstractmethod
    def compute(self, structure: Structure) -> Structure:
        """
        Computes properties (energy, forces, stress) for a single structure.

        Args:
            structure: The structure to compute.

        Returns:
            Structure: The structure with computed properties attached.
        """
        pass

    @abstractmethod
    def compute_batch(self, structures: Iterator[Structure]) -> Iterator[Structure]:
        """
        Computes properties for a batch of structures.
        Must return an iterator to allow streaming.

        Args:
            structures: An iterator of structures to compute.

        Returns:
            Iterator[Structure]: A stream of computed structures.
        """
        pass

class BaseTrainer(BaseComponent):
    """
    Abstract base class for MLIP trainers.
    Responsible for fitting interatomic potentials to the labeled dataset.
    """

    @abstractmethod
    def train(
        self,
        dataset: Dataset,
        initial_potential: Optional[Potential] = None
    ) -> Potential:
        """
        Trains a potential on the given dataset.

        Args:
            dataset: The labeled dataset (abstraction, not list).
            initial_potential: Optional starting potential for fine-tuning.

        Returns:
            Potential: The trained potential artifact (e.g., path to file).
        """
        pass

class BaseDynamics(BaseComponent):
    """
    Abstract base class for dynamics engines (MD/kMC).
    Responsible for running simulations and exploring phase space.
    """

    @abstractmethod
    def explore(
        self,
        potential: Potential,
        initial_structure: Structure,
        cycle: int = 0,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Runs exploration using the potential.

        Args:
            potential: The potential to use for forces/energy.
            initial_structure: Starting configuration.
            cycle: Current active learning cycle number.
            metrics: Optional metrics to guide exploration settings.

        Returns:
            Dict[str, Any]: Metrics and results from the exploration (e.g., 'halted', 'structures').
        """
        pass

class BaseValidator(BaseComponent):
    """
    Abstract base class for validators.
    Responsible for assessing the quality and physical correctness of the potential.
    """

    @abstractmethod
    def validate(self, potential: Potential) -> Dict[str, Any]:
        """
        Validates the potential against test set and physical constraints.

        Args:
            potential: The potential to validate.

        Returns:
            Dict[str, Any]: Validation metrics (e.g., RMSE, stability flags).
        """
        pass
