"""Structure Generator module implementation."""

from collections.abc import Iterable, Iterator
from typing import Any

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.interfaces import StructureGenerator
from pyacemaker.core.utils import generate_dummy_structures
from pyacemaker.domain_models.models import StructureMetadata, StructureStatus
from pyacemaker.generator.policy import AdaptivePolicy, ExplorationContext


class BaseStructureGenerator(StructureGenerator):
    """Base class for structure generators with shared logic."""

    def run(self) -> ModuleResult:
        """Run the main structure generation workflow."""
        self.logger.info(f"Running {self.__class__.__name__}")
        return ModuleResult(
            status="success",
            metrics=Metrics.model_validate({"generated_count": 0}),
        )

    def generate_batch_candidates(
        self,
        seed_structures: Iterable[StructureMetadata],
        n_candidates_per_seed: int,
        cycle: int = 0,
    ) -> Iterator[StructureMetadata]:
        """Generate candidate structures for a batch of seeds."""
        if n_candidates_per_seed <= 0:
            msg = "Number of candidates per seed must be positive"
            raise ValueError(msg)

        self.logger.info(f"Generating batch candidates ({self.__class__.__name__})")

        for seed in seed_structures:
            yield from self.generate_local_candidates(seed, n_candidates_per_seed, cycle=cycle)


class RandomStructureGenerator(BaseStructureGenerator):
    """Generates random structures for testing."""

    def generate_initial_structures(self) -> Iterator[StructureMetadata]:
        """Generate initial structures."""
        self.logger.info("Generating initial structures (mock)")
        yield from generate_dummy_structures(5, tags=["initial", "random"])

    def _generate_candidates_common(
        self, n_candidates: int, tags: list[str]
    ) -> Iterator[StructureMetadata]:
        """Generate candidate structures (common logic)."""
        yield from generate_dummy_structures(n_candidates, tags=tags)

    def generate_local_candidates(
        self, seed_structure: StructureMetadata, n_candidates: int, cycle: int = 0
    ) -> Iterator[StructureMetadata]:
        """Generate local candidates."""
        if seed_structure is None:
            msg = "Seed structure cannot be None"
            raise ValueError(msg)
        if n_candidates <= 0:
            msg = "Number of candidates must be positive"
            raise ValueError(msg)

        self.logger.info(f"Generating {n_candidates} local candidates around {seed_structure.id}")
        yield from self._generate_candidates_common(n_candidates, tags=["candidate", "local"])

    def get_strategy_info(self) -> dict[str, Any]:
        """Return information about the current exploration strategy."""
        return {"strategy": "random", "parameters": {}}


class AdaptiveStructureGenerator(BaseStructureGenerator):
    """Generates structures using an adaptive exploration policy based on material features."""

    def __init__(self, config: PYACEMAKERConfig) -> None:
        """Initialize."""
        super().__init__(config)
        self.policy = AdaptivePolicy(config)

    def generate_initial_structures(self) -> Iterator[StructureMetadata]:
        """Generate initial structures using Cold Start policy."""
        self.logger.info("Generating initial structures (Adaptive Cold Start)")

        context = ExplorationContext(cycle=0)
        strategy = self.policy.decide_strategy(context)

        # Create base prototypes directly using ASE to avoid overhead
        try:
            from ase.build import bulk
        except ImportError:
            self.logger.warning("ASE not found. Using dummy generation.")
            yield from generate_dummy_structures(5, tags=["initial", "fallback"])
            return

        # Simple set of prototypes for cold start
        # In a real app, this would come from a config or database
        elements = ["Fe", "Al", "Cu", "Si", "C"]

        for el in elements:
            try:
                atoms = bulk(el, cubic=True)
                # Apply strategy to diversify
                candidates = strategy.generate(atoms, n_candidates=1)
                for cand_atoms in candidates:
                    yield StructureMetadata(
                        features={"atoms": cand_atoms},
                        tags=["initial", "adaptive", f"strategy:{type(strategy).__name__}"],
                        status=StructureStatus.NEW,
                    )
            except Exception:
                self.logger.warning(f"Failed to generate prototype for {el}")
                continue

    def generate_local_candidates(
        self, seed_structure: StructureMetadata, n_candidates: int, cycle: int = 0
    ) -> Iterator[StructureMetadata]:
        """Generate local candidates using adaptive policy."""
        if seed_structure is None:
            msg = "Seed structure cannot be None"
            raise ValueError(msg)
        if n_candidates <= 0:
            msg = "Number of candidates must be positive"
            raise ValueError(msg)

        context = ExplorationContext(
            cycle=cycle,
            seed_structure=seed_structure,
            uncertainty_state=seed_structure.uncertainty_state,
        )
        strategy = self.policy.decide_strategy(context)

        self.logger.info(
            f"Generating {n_candidates} candidates for {seed_structure.id} with {type(strategy).__name__}"
        )

        atoms = seed_structure.features.get("atoms")
        if not atoms:
            self.logger.warning(
                f"Seed structure {seed_structure.id} has no 'atoms' feature. Skipping."
            )
            return

        candidates = strategy.generate(atoms, n_candidates=n_candidates)

        for cand_atoms in candidates:
            yield StructureMetadata(
                features={"atoms": cand_atoms},
                tags=["candidate", "adaptive", f"strategy:{type(strategy).__name__}"],
                status=StructureStatus.NEW,
            )

    def get_strategy_info(self) -> dict[str, Any]:
        """Return information about the current exploration strategy."""
        return {"strategy": "adaptive", "parameters": self.config.structure_generator.model_dump()}
