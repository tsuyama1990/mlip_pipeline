"""Direct Generator implementation."""

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig
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
        """Generate structures using MaxMin diversity sampling."""
        self.logger.info(f"Generating {n_samples} samples using MaxMin diversity ({objective})")

        # Oversampling factor
        oversample = int(self.config.structure_generator.parameters.get("oversample", 10))
        n_candidates = n_samples * oversample

        # Generate pool of random candidates
        # Using dummy structures with random positions for now
        # In real scenario, this would use a proper random structure generator
        candidates: list[StructureMetadata] = []

        # We need actual diverse positions for the test to pass diversity check
        raw_candidates = list(generate_dummy_structures(n_candidates, tags=["pool"]))

        # Get generation parameters from config
        box_size = float(self.config.structure_generator.parameters.get("box_size", 10.0))

        # Randomize positions
        for _i, s in enumerate(raw_candidates):
            atoms = s.features.get("atoms")
            if atoms:
                # Randomize positions in a box
                n_atoms = len(atoms)
                new_pos = np.random.rand(n_atoms, 3) * box_size
                atoms.set_positions(new_pos)
                s.features["atoms"] = atoms
            candidates.append(s)

        # Compute descriptors (simple: sorted pairwise distances)
        # Using flat positions for simplicity as descriptor
        descriptors = []
        for s in candidates:
            atoms = s.features.get("atoms")
            if atoms:
                # Flatten positions as simple descriptor
                descriptors.append(atoms.get_positions().flatten())
            else:
                descriptors.append(np.zeros(1))

        descriptors_array = np.array(descriptors)

        # MaxMin Selection
        selected_indices = [0] # Start with first
        selected_descriptors = [descriptors_array[0]]

        for _ in range(n_samples - 1):
            # Calculate distance from remaining candidates to selected set
            # We want to maximize the minimum distance to any selected point

            # Distance matrix between all candidates and selected
            # Optimization: only compute for non-selected
            remaining_indices = [i for i in range(n_candidates) if i not in selected_indices]
            if not remaining_indices:
                break

            # Compute min distance to ANY selected point for each candidate
            # d_min(x) = min_{y in selected} dist(x, y)
            # We want x that maximizes d_min(x)

            # Using cdist
            dists = cdist(descriptors_array[remaining_indices], np.array(selected_descriptors))
            min_dists = np.min(dists, axis=1)

            # Select candidate with max min_dist
            best_idx_in_remaining = np.argmax(min_dists)
            best_idx = remaining_indices[best_idx_in_remaining]

            selected_indices.append(best_idx)
            selected_descriptors.append(descriptors_array[best_idx])

        # Yield selected
        for idx in selected_indices:
            s = candidates[idx]
            s.generation_method = "direct"
            s.tags = ["initial", "direct", f"objective:{objective}"]
            yield s

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
