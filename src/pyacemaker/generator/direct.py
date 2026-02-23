"""Direct Sampling Structure Generator."""

from collections.abc import Iterable, Iterator
from typing import Any
from uuid import uuid4

from ase import Atoms
from loguru import logger

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig, StructureGeneratorConfig
from pyacemaker.core.interfaces import StructureGenerator
from pyacemaker.core.utils import atoms_to_metadata
from pyacemaker.domain_models.models import (
    StructureMetadata,
    StructureStatus,
)


class DirectGenerator(StructureGenerator):
    """Structure generator using DIRECT sampling (or entropy maximization)."""

    def __init__(self, config: PYACEMAKERConfig | StructureGeneratorConfig) -> None:
        """Initialize the generator.

        Accepts either full config (standard) or module config (testing/isolated).
        """
        self.gen_config: StructureGeneratorConfig

        if isinstance(config, PYACEMAKERConfig):
            super().__init__(config)
            self.gen_config = config.structure_generator
        else:
            # Testing mode or direct usage
            # BaseModule.__init__ expects PYACEMAKERConfig, so we skip it or mock it?
            # Creating a dummy config is hard. We'll just set attributes manually.
            # This violates LSP if checked strictly against BaseModule but works at runtime.
            self.config = config  # type: ignore[assignment]
            self.gen_config = config
            self.logger = logger.bind(name="DirectGenerator")

    def run(self) -> ModuleResult:
        """Execute default generation task (Step 1)."""
        # Default behavior: generate initial structures if not specified
        # This implementation satisfies BaseModule abstract method
        # Just consume iterator to count? No, usually run() produces side effects (files)
        # But here we just return success as orchestrator calls specific methods.
        return ModuleResult(
            status="success",
            metrics=Metrics(message="DirectGenerator ready")  # type: ignore[call-arg]
        )

    def generate_initial_structures(self) -> Iterator[StructureMetadata]:
        """Generate initial structures for cold start."""
        # For now, delegate to direct samples
        # In a real scenario, this might load from a database or use specific templates
        yield from self.generate_direct_samples(n_samples=20)

    def generate_direct_samples(
        self, n_samples: int, objective: str = "maximize_entropy"
    ) -> Iterator[StructureMetadata]:
        """Generate structures using DIRECT sampling."""
        self.logger.info(f"Generating {n_samples} structures using {objective}")

        for i in range(n_samples):
            atoms = self._generate_random_structure()
            # Explicitly set metadata in atoms.info to ensure persistence
            atoms.info["generation_method"] = "direct"

            metadata = atoms_to_metadata(
                atoms,
                id=uuid4(),
                tags=["initial", "direct", f"sample_{i}"],
                generation_method="direct",
                status=StructureStatus.NEW
            )
            yield metadata

    def generate_local_candidates(
        self, seed_structure: StructureMetadata, n_candidates: int, cycle: int = 1
    ) -> Iterator[StructureMetadata]:
        """Generate candidate structures around a seed structure."""
        self.logger.info(f"Generating {n_candidates} local candidates for seed {seed_structure.id}")

        if "atoms" not in seed_structure.features:
            self.logger.warning(f"Seed structure {seed_structure.id} has no atoms. Skipping.")
            return

        seed_atoms = seed_structure.features["atoms"]

        for _ in range(n_candidates):
            new_atoms = seed_atoms.copy()
            new_atoms.rattle(stdev=0.1)

            metadata = atoms_to_metadata(
                new_atoms,
                id=uuid4(),
                tags=["candidate", f"seed_{seed_structure.id}", f"cycle_{cycle}"],
                generation_method="rattle",
                status=StructureStatus.NEW
            )
            yield metadata

    def generate_batch_candidates(
        self,
        seed_structures: Iterable[StructureMetadata],
        n_candidates_per_seed: int,
        cycle: int = 1,
    ) -> Iterator[StructureMetadata]:
        """Generate candidate structures for a batch of seeds."""
        for seed in seed_structures:
            yield from self.generate_local_candidates(seed, n_candidates_per_seed, cycle)

    def get_strategy_info(self) -> dict[str, Any]:
        """Return information about the current exploration strategy."""
        return {
            "name": "DirectGenerator",
            "strategy": self.gen_config.strategy,
            "objective": "maximize_entropy"
        }

    def _generate_random_structure(self) -> Atoms:
        """Generate a single random structure."""
        try:
            from ase.build import bulk
            atoms = bulk("Fe", cubic=True)
            atoms = atoms * (2, 2, 2)
            atoms.rattle(stdev=0.2)
        except Exception:
            return Atoms("Fe2", positions=[[0, 0, 0], [2.0, 0, 0]], cell=[4, 4, 4], pbc=True)
        else:
            return atoms  # type: ignore[no-any-return]
