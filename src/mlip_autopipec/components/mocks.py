from collections.abc import Iterator

from mlip_autopipec.components.base import (
    BaseDynamics,
    BaseGenerator,
    BaseOracle,
    BaseTrainer,
    BaseValidator,
)
from mlip_autopipec.config import (
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


class MockGenerator(BaseGenerator):
    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__(config)

    def generate(self, n_structures: int) -> Iterator[Structure]:
        """Yields an empty sequence of structures."""
        return iter([])


class MockOracle(BaseOracle):
    def __init__(self, config: OracleConfig) -> None:
        super().__init__(config)

    def compute(self, structures: Iterator[Structure]) -> Iterator[CalculationResult]:
        """Consumes structures and yields empty results."""
        # Consume iterator to prevent "unused" warnings in real scenarios
        for _ in structures:
            pass
        return iter([])


class MockTrainer(BaseTrainer):
    def __init__(self, config: TrainerConfig) -> None:
        super().__init__(config)

    def train(
        self,
        dataset: Iterator[CalculationResult],
        previous_potential: PotentialArtifact | None = None,
    ) -> PotentialArtifact:
        """Consumes dataset and returns a mock potential."""
        # Consume iterator
        for _ in dataset:
            pass
        return PotentialArtifact(path="mock.yace", version="0.1")


class MockDynamics(BaseDynamics):
    def __init__(self, config: DynamicsConfig) -> None:
        super().__init__(config)

    def explore(self, potential: PotentialArtifact) -> ExplorationResult:
        return ExplorationResult(total_steps=0, halt_count=0, exploration_time=0.0)


class MockValidator(BaseValidator):
    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config)

    def validate(self, potential: PotentialArtifact) -> bool:
        return True
