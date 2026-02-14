"""Structure Generator module implementation."""

from collections.abc import Iterable, Iterator
from typing import Any

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.interfaces import StructureGenerator
from pyacemaker.core.utils import generate_dummy_structures
from pyacemaker.domain_models.models import StructureMetadata


class RandomStructureGenerator(StructureGenerator):
    """Generates random structures for testing."""

    def __init__(self, config: PYACEMAKERConfig) -> None:
        """Initialize."""
        super().__init__(config)

    def run(self) -> ModuleResult:
        """Run the main structure generation workflow."""
        self.logger.info("Running StructureGenerator")
        # In a real scenario, this might explore and return a result summary
        # For 'run', we might still count them, but be careful with large streams
        # Here we just iterate and count to simulate work without keeping all in memory
        count = sum(1 for _ in self.generate_initial_structures())

        return ModuleResult(
            status="success",
            metrics=Metrics.model_validate({"generated_count": count}),
        )

    def generate_initial_structures(self) -> Iterator[StructureMetadata]:
        """Generate initial structures."""
        self.logger.info("Generating initial structures (mock)")
        # Return iterator directly
        yield from generate_dummy_structures(5, tags=["initial", "random"])

    def generate_local_candidates(
        self, seed_structure: StructureMetadata, n_candidates: int
    ) -> Iterator[StructureMetadata]:
        """Generate local candidates."""
        if seed_structure is None:
            msg = "Seed structure cannot be None"
            raise ValueError(msg)
        if n_candidates <= 0:
            msg = "Number of candidates must be positive"
            raise ValueError(msg)

        self.logger.info(f"Generating {n_candidates} local candidates around {seed_structure.id}")
        yield from generate_dummy_structures(n_candidates, tags=["candidate", "local"])

    def generate_batch_candidates(
        self, seed_structures: Iterable[StructureMetadata], n_candidates_per_seed: int
    ) -> Iterator[StructureMetadata]:
        """Generate candidate structures for a batch of seeds."""
        if n_candidates_per_seed <= 0:
            msg = "Number of candidates per seed must be positive"
            raise ValueError(msg)

        # We can't know total count upfront for logs if it's an iterator
        self.logger.info("Generating batch candidates from seed structures")

        # Use generator to yield candidates one by one or in small batches
        # We process seeds one by one to keep memory low
        for _ in seed_structures:
            yield from generate_dummy_structures(n_candidates_per_seed, tags=["candidate", "batch"])

    def get_strategy_info(self) -> dict[str, Any]:
        """Return information about the current exploration strategy."""
        return {"strategy": "random", "parameters": {}}
