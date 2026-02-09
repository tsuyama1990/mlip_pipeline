from abc import ABC, abstractmethod
from collections.abc import Iterator

from mlip_autopipec.config import (
    BaseComponentConfig,
    DynamicsConfig,
    GeneratorConfig,
    OracleConfig,
    TrainerConfig,
    ValidatorConfig,
)
from mlip_autopipec.domain_models.inputs import Structure
from mlip_autopipec.domain_models.results import (
    CalculationResult,
    ExplorationResult,
    PotentialArtifact,
)


class BaseComponent:
    """Base class for all pipeline components."""

    def __init__(self, config: BaseComponentConfig) -> None:
        self.config = config


class BaseGenerator(BaseComponent, ABC):
    """Generates initial structures or candidates."""

    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__(config)
        self.config: GeneratorConfig = config  # Type narrowing

    @abstractmethod
    def generate(self, n_structures: int) -> Iterator[Structure]:
        """Generate a stream of structures."""


class BaseOracle(BaseComponent, ABC):
    """Performs first-principles calculations."""

    def __init__(self, config: OracleConfig) -> None:
        super().__init__(config)
        self.config: OracleConfig = config

    @abstractmethod
    def compute(self, structures: Iterator[Structure]) -> Iterator[CalculationResult]:
        """Compute properties for a stream of structures."""


class BaseTrainer(BaseComponent, ABC):
    """Trains the machine learning potential."""

    def __init__(self, config: TrainerConfig) -> None:
        super().__init__(config)
        self.config: TrainerConfig = config

    @abstractmethod
    def train(
        self,
        dataset: Iterator[CalculationResult],
        previous_potential: PotentialArtifact | None = None,
    ) -> PotentialArtifact:
        """Train a potential on the dataset."""


class BaseDynamics(BaseComponent, ABC):
    """Runs molecular dynamics or sampling simulations."""

    def __init__(self, config: DynamicsConfig) -> None:
        super().__init__(config)
        self.config: DynamicsConfig = config

    @abstractmethod
    def explore(self, potential: PotentialArtifact) -> ExplorationResult:
        """Run exploration using the potential."""


class BaseValidator(BaseComponent, ABC):
    """Validates the quality of the potential."""

    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config)
        self.config: ValidatorConfig = config

    @abstractmethod
    def validate(self, potential: PotentialArtifact) -> bool:
        """Validate the potential. Returns True if valid."""
