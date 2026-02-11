import abc
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.inputs import ProjectState, Structure
from mlip_autopipec.domain_models.results import TrainingResult


class BaseComponent(abc.ABC):
    """Base class for all pipeline components."""

    def __init__(self, config: Any, work_dir: Path) -> None:
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def _validate_config(self) -> None:
        """Validate component configuration."""


class BaseGenerator(BaseComponent):
    """Abstract base class for structure generators."""

    @abc.abstractmethod
    def generate(self, state: ProjectState) -> Iterator[Structure]:
        """Generate global candidate structures (e.g., for Cold Start)."""

    @abc.abstractmethod
    def generate_local(self, input_structure: Structure, n_candidates: int) -> Iterator[Structure]:
        """Generate local candidates around an input structure (e.g., for Halt Recovery)."""


class BaseOracle(BaseComponent):
    """Abstract base class for Oracles (DFT/Calculators)."""

    @abc.abstractmethod
    def compute(self, structures: Iterator[Structure]) -> Iterator[Structure]:
        """Compute properties for structures (in-place or new objects)."""


class BaseTrainer(BaseComponent):
    """Abstract base class for Potential Trainers."""

    @abc.abstractmethod
    def train(self, dataset_path: Path, previous_potential: Path | None = None) -> TrainingResult:
        """Train a potential on the given dataset."""

    @abc.abstractmethod
    def select_local_active_set(self, candidates: Iterator[Structure], n_selection: int) -> Iterator[Structure]:
        """Select the most informative structures from a candidate set (D-Optimality)."""


class BaseDynamics(BaseComponent):
    """Abstract base class for Dynamics/Exploration engines."""

    @abc.abstractmethod
    def explore(self, potential_path: Path, initial_structure: Structure) -> Iterator[Structure]:
        """Run dynamics and yield structures (e.g., halts)."""


class BaseValidator(BaseComponent):
    """Abstract base class for Validators."""

    @abc.abstractmethod
    def validate(self, potential_path: Path) -> dict[str, Any]:
        """Run validation tests and return metrics."""
