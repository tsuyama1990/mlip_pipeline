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
        # Just return success, don't consume the generator for counting as it's wasteful
        return ModuleResult(
            status="success",
            metrics=Metrics.model_validate({"generated_count": 0}),  # Placeholder
        )

    def generate_initial_structures(self) -> Iterator[StructureMetadata]:
        """Generate initial structures."""
        self.logger.info("Generating initial structures (mock)")
        # Return iterator directly
        yield from generate_dummy_structures(5, tags=["initial", "random"])

    def _generate_candidates_common(
        self, n_candidates: int, tags: list[str]
    ) -> Iterator[StructureMetadata]:
        """Generate candidate structures (common logic)."""
        yield from generate_dummy_structures(n_candidates, tags=tags)

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
        yield from self._generate_candidates_common(n_candidates, tags=["candidate", "local"])

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
            yield from self._generate_candidates_common(
                n_candidates_per_seed, tags=["candidate", "batch"]
            )

    def get_strategy_info(self) -> dict[str, Any]:
        """Return information about the current exploration strategy."""
        return {"strategy": "random", "parameters": {}}


class AdaptiveStructureGenerator(StructureGenerator):
    """Generates structures using an adaptive exploration policy based on material features."""

    def __init__(self, config: PYACEMAKERConfig) -> None:
        """Initialize."""
        super().__init__(config)

    def run(self) -> ModuleResult:
        """Run the main structure generation workflow."""
        self.logger.info("Running AdaptiveStructureGenerator")
        return ModuleResult(
            status="success",
            metrics=Metrics.model_validate({"generated_count": 0}),  # Placeholder
        )

    def _determine_policy(self, structure: StructureMetadata) -> dict[str, Any]:
        """Determine exploration policy based on structure features."""
        # Load rules from configuration parameters, or use defaults
        rules = self.config.structure_generator.parameters.get("policy_rules", {})

        # Default policy
        policy: dict[str, Any] = {
            "mode": rules.get("default_mode", "default"),
            "temperature": rules.get("default_temp", 300.0)
        }

        # Check Predicted Properties (e.g., Band Gap -> Metal/Insulator logic)
        if structure.predicted_properties:
            bg = structure.predicted_properties.band_gap
            # If band gap > 0 (insulator), apply defect strategy
            if bg is not None and bg > rules.get("insulator_band_gap_threshold", 0.0):
                policy["mode"] = "defect_driven"
                policy["n_defects"] = rules.get("insulator_defects", 2)
            else:
                policy["mode"] = "high_mc"
                policy["mc_ratio"] = rules.get("metal_mc_ratio", 0.5)

        # Check Uncertainty (High uncertainty -> Cautious exploration)
        if structure.uncertainty_state:
            gamma_max = structure.uncertainty_state.gamma_max
            gamma_thresh = rules.get("high_uncertainty_threshold", 5.0)
            if gamma_max is not None and gamma_max > gamma_thresh:
                policy["mode"] = "cautious"
                policy["temperature"] = rules.get("cautious_temp", 100.0)

        return policy

    def generate_initial_structures(self) -> Iterator[StructureMetadata]:
        """Generate initial structures based on adaptive strategy (Mock)."""
        self.logger.info("Generating initial structures (Adaptive)")
        # In a real scenario, this would use M3GNet/Universal Potentials for cold start
        yield from generate_dummy_structures(5, tags=["initial", "adaptive", "cold_start"])

    def generate_local_candidates(
        self, seed_structure: StructureMetadata, n_candidates: int
    ) -> Iterator[StructureMetadata]:
        """Generate local candidates using adaptive policy."""
        if seed_structure is None:
            msg = "Seed structure cannot be None"
            raise ValueError(msg)
        if n_candidates <= 0:
            msg = "Number of candidates must be positive"
            raise ValueError(msg)

        policy = self._determine_policy(seed_structure)
        self.logger.info(
            f"Generating {n_candidates} candidates for {seed_structure.id} with policy {policy['mode']}"
        )

        tags = ["candidate", "adaptive", f"policy:{policy['mode']}"]
        yield from generate_dummy_structures(n_candidates, tags=tags)

    def generate_batch_candidates(
        self, seed_structures: Iterable[StructureMetadata], n_candidates_per_seed: int
    ) -> Iterator[StructureMetadata]:
        """Generate candidate structures for a batch of seeds."""
        if n_candidates_per_seed <= 0:
            msg = "Number of candidates per seed must be positive"
            raise ValueError(msg)

        self.logger.info("Generating batch candidates (Adaptive)")

        for seed in seed_structures:
            yield from self.generate_local_candidates(seed, n_candidates_per_seed)

    def get_strategy_info(self) -> dict[str, Any]:
        """Return information about the current exploration strategy."""
        return {"strategy": "adaptive", "parameters": {"policy_logic": "mock_rules"}}
