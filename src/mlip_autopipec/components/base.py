from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from mlip_autopipec.domain_models import ComponentConfig, Dataset, Potential, Structure


class BaseComponent:
    """
    Base class for all pipeline components.
    """
    def __init__(self, config: ComponentConfig) -> None:
        self.config = config

class BaseGenerator(BaseComponent, ABC):
    """
    Abstract base class for structure generators.
    """
    @abstractmethod
    def generate(self, n_structures: int, cycle: int = 0, metrics: dict[str, Any] | None = None) -> Iterator[Structure]:
        """
        Generates candidate structures.
        """

class BaseOracle(BaseComponent, ABC):
    """
    Abstract base class for oracles (calculators).
    """
    @abstractmethod
    def compute(self, structures: Iterator[Structure]) -> Iterator[Structure]:
        """
        Computes properties (energy, forces, stress) for the given structures.
        """

class BaseTrainer(BaseComponent, ABC):
    """
    Abstract base class for potential trainers.
    """
    @abstractmethod
    def train(self, dataset: Dataset) -> Potential:
        """
        Trains a potential on the given dataset.
        """
