"""Direct Generator implementation."""

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig
from pyacemaker.core.interfaces import StructureGenerator
from pyacemaker.core.utils import generate_dummy_structures
from pyacemaker.domain_models.models import StructureMetadata, StructureStatus


class DirectGenerator(StructureGenerator):
    """Generates structures using Direct Sampling (MaxMin Diversity)."""

    def __init__(self, config: PYACEMAKERConfig) -> None:
        """Initialize."""
        super().__init__(config)
        self.config = config

    def run(self) -> ModuleResult:
        """Run the main structure generation workflow."""
        self.logger.info("Running DirectGenerator")
        return ModuleResult(
            status="success",
            metrics=Metrics.model_validate({"generated_count": 0}),  # Placeholder
        )

    def generate_initial_structures(self) -> Iterator[StructureMetadata]:
        """Generate initial structures (Mock/Random)."""
        # For initial structures, we can just use random
        self.logger.info("Generating initial structures (Direct/Random)")
        yield from generate_dummy_structures(20, tags=["initial", "direct"])

    def _get_next_valid_candidate(
        self, gen: Iterator[StructureMetadata]
    ) -> tuple[StructureMetadata | None, np.ndarray | None]:
        """Get next valid candidate with atoms feature."""
        try:
            while True:
                s = next(gen)
                atoms = s.features.get("atoms")
                if atoms:
                    desc = atoms.get_positions().flatten()
                    return s, desc
        except StopIteration:
            return None, None

    def _collect_batch(
        self,
        gen: Iterator[StructureMetadata],
        oversample: int,
        ref_shape: tuple[int, ...],
        box_size: float
    ) -> tuple[list[StructureMetadata], list[np.ndarray]]:
        """Collect a batch of valid candidates."""
        batch_candidates: list[StructureMetadata] = []
        batch_descriptors: list[np.ndarray] = []

        # We assume gen yields structures with randomized positions already
        # but if we needed to randomize here, we could.
        # The generator logic handles it.

        candidates_checked = 0
        while candidates_checked < oversample:
            s, desc = self._get_next_valid_candidate(gen)
            if s is None:
                break

            candidates_checked += 1
            if desc.shape == ref_shape:
                batch_candidates.append(s)
                batch_descriptors.append(desc)

        return batch_candidates, batch_descriptors

    def generate_direct_samples(
        self, n_samples: int, objective: str = "maximize_entropy"
    ) -> Iterator[StructureMetadata]:
        """Generate structures using Batched MaxMin diversity sampling."""
        self.logger.info(f"Generating {n_samples} samples using Batched MaxMin diversity ({objective})")

        oversample = int(
            self.config.structure_generator.parameters.get(
                "oversample", CONSTANTS.direct_oversample
            )
        )

        box_size = float(
            self.config.structure_generator.parameters.get(
                "box_size", CONSTANTS.direct_box_size
            )
        )

        def candidate_generator() -> Iterator[StructureMetadata]:
            total_needed = n_samples * oversample
            base_gen = generate_dummy_structures(total_needed, tags=["pool"])
            for s in base_gen:
                atoms = s.features.get("atoms")
                if atoms:
                    n_atoms = len(atoms)
                    new_pos = np.random.rand(n_atoms, 3) * box_size
                    atoms.set_positions(new_pos)
                    s.features["atoms"] = atoms
                yield s

        candidates_gen = candidate_generator()
        selected_descriptors: list[np.ndarray] = []

        # 1. Select first point
        first_valid, first_desc = self._get_next_valid_candidate(candidates_gen)

        if first_valid is None:
            return

        s = first_valid
        s.generation_method = "direct"
        s.tags = ["initial", "direct", f"objective:{objective}"]
        selected_descriptors.append(first_desc)
        yield s

        # 2. Iteratively select remaining
        for _ in range(n_samples - 1):
            batch_candidates, batch_descriptors = self._collect_batch(
                candidates_gen, oversample, selected_descriptors[0].shape, box_size
            )

            if not batch_candidates:
                break

            batch_desc_arr = np.array(batch_descriptors)
            min_dists = np.full(len(batch_candidates), np.inf)

            chunk_size = 1000
            for i in range(0, len(selected_descriptors), chunk_size):
                chunk = selected_descriptors[i : i + chunk_size]
                chunk_arr = np.array(chunk)
                dists_chunk = cdist(batch_desc_arr, chunk_arr)
                chunk_mins = np.min(dists_chunk, axis=1)
                min_dists = np.minimum(min_dists, chunk_mins)

            best_idx = np.argmax(min_dists)
            best_s = batch_candidates[best_idx]
            best_desc = batch_descriptors[best_idx]

            best_s.generation_method = "direct"
            best_s.tags = ["initial", "direct", f"objective:{objective}"]

            selected_descriptors.append(best_desc)
            yield best_s

    def generate_local_candidates(
        self, seed_structure: StructureMetadata, n_candidates: int, cycle: int = 1
    ) -> Iterator[StructureMetadata]:
        """Generate local candidates (Simple Perturbation)."""
        # Fallback implementation
        self.logger.info(f"Generating local candidates around {seed_structure.id}")

        atoms = seed_structure.features.get("atoms")
        if not atoms:
            return

        rattle_amp = self.config.structure_generator.rattle_amplitude
        perturbation_scale = rattle_amp * 2.0  # Range [-rattle, +rattle]

        for _ in range(n_candidates):
            # Perturb positions
            new_atoms = atoms.copy()
            pos = new_atoms.get_positions()

            # Use configured rattle amplitude
            pos += (np.random.rand(*pos.shape) - 0.5) * perturbation_scale
            new_atoms.set_positions(pos)

            yield StructureMetadata(
                features={"atoms": new_atoms},
                tags=["candidate", "local", f"seed:{seed_structure.id}"],
                status=StructureStatus.NEW
            )

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
        """Return strategy info."""
        return {"strategy": "direct_maxmin", "parameters": {}}
