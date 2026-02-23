"""Direct Generator implementation."""

from collections.abc import Iterable, Iterator
from itertools import islice
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

    def generate_direct_samples(
        self, n_samples: int, objective: str = "maximize_entropy"
    ) -> Iterator[StructureMetadata]:
        """Generate structures using Batched MaxMin diversity sampling.

        Uses a Batched Greedy approach to avoid O(N) memory usage for candidates.
        """
        self.logger.info(f"Generating {n_samples} samples using Batched MaxMin diversity ({objective})")

        # Oversampling factor determines batch size relative to 1 selection
        # For each selection, we look at 'oversample' candidates
        oversample = int(
            self.config.structure_generator.parameters.get(
                "oversample", CONSTANTS.direct_oversample
            )
        )
        # Batch size for distance computation (chunking)
        batch_size = max(CONSTANTS.direct_batch_size, oversample)

        box_size = float(
            self.config.structure_generator.parameters.get(
                "box_size", CONSTANTS.direct_box_size
            )
        )

        # Generator for randomized candidates
        def candidate_generator() -> Iterator[StructureMetadata]:
            # Infinite stream effectively, or large enough
            # We need n_samples * oversample total candidates approximately
            total_needed = n_samples * oversample

            # Use generate_dummy_structures as base
            base_gen = generate_dummy_structures(total_needed, tags=["pool"])

            for s in base_gen:
                atoms = s.features.get("atoms")
                if atoms:
                    # Randomize positions
                    n_atoms = len(atoms)
                    new_pos = np.random.rand(n_atoms, 3) * box_size
                    atoms.set_positions(new_pos)
                    s.features["atoms"] = atoms
                yield s

        candidates_gen = candidate_generator()

        # Selected set (descriptors)
        # Memory usage scales with n_samples (O(K * D)), not total candidates.
        selected_descriptors: list[np.ndarray] = []

        # 1. Select first point randomly (or first valid from stream)
        try:
            first_valid = None
            first_desc = None

            while first_valid is None:
                s = next(candidates_gen)
                atoms = s.features.get("atoms")
                if atoms:
                    desc = atoms.get_positions().flatten()
                    first_valid = s
                    first_desc = desc

            s = first_valid
            s.generation_method = "direct"
            s.tags = ["initial", "direct", f"objective:{objective}"]
            selected_descriptors.append(first_desc)
            yield s

        except StopIteration:
            return

        # 2. Iteratively select remaining
        # For each needed sample, we process 'oversample' candidates (or a batch)
        # and pick the one maximizing min-dist to selected_descriptors.

        for _ in range(n_samples - 1):
            # Consume a batch of candidates
            batch_candidates: list[StructureMetadata] = []
            batch_descriptors: list[np.ndarray] = []

            # We want to scan 'oversample' candidates to find the next best one
            # But we must ensure consistent shape

            candidates_checked = 0
            while candidates_checked < oversample:
                try:
                    s = next(candidates_gen)
                except StopIteration:
                    break

                candidates_checked += 1
                atoms = s.features.get("atoms")
                if atoms:
                    desc = atoms.get_positions().flatten()
                    # Check shape consistency
                    if desc.shape == selected_descriptors[0].shape:
                        batch_candidates.append(s)
                        batch_descriptors.append(desc)

            if not batch_candidates:
                break

            # Compute distances
            # (Batch, Selected)
            batch_desc_arr = np.array(batch_descriptors)
            selected_desc_arr = np.array(selected_descriptors)

            dists = cdist(batch_desc_arr, selected_desc_arr)
            # min dist to any selected
            min_dists = np.min(dists, axis=1)

            # Maximize min dist
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
