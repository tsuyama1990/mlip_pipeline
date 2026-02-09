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
        # For testing memory safety, we should be able to yield structures lazily.
        # But Mock usually just returns nothing or a list.
        # Let's yield a few dummy structures if n_structures > 0 to simulate work
        # But for now, returning iter([]) is safe as per original logic.
        return iter([])


class MockOracle(BaseOracle):
    def __init__(self, config: OracleConfig) -> None:
        super().__init__(config)

    def compute(self, structures: Iterator[Structure]) -> Iterator[CalculationResult]:
        """Consumes structures and yields mock results."""
        for structure in structures:
            # Yield a result for each structure
            # In a real scenario, this would compute properties
            yield CalculationResult(
                structure=structure,
                energy=-10.0,
                forces=[[0.0, 0.0, 0.0] for _ in range(len(structure.atoms))],
                stress=[0.0] * 6,
                converged=True,
            )


class MockTrainer(BaseTrainer):
    def __init__(self, config: TrainerConfig) -> None:
        super().__init__(config)

    def train(
        self,
        dataset: Iterator[CalculationResult],
        previous_potential: PotentialArtifact | None = None,
    ) -> PotentialArtifact:
        """Consumes dataset and returns a mock potential."""
        # Consume iterator to ensure stream processing is possible
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
