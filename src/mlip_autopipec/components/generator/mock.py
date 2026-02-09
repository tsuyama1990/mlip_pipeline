import logging
from collections.abc import Iterator
from typing import Any

import numpy as np

from mlip_autopipec.components.generator.base import BaseGenerator
from mlip_autopipec.domain_models.config import MockGeneratorConfig
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class MockGenerator(BaseGenerator):
    """
    Mock implementation of the Generator component.

    Generates random structures based on configured parameters (cell size, number of atoms, etc.).
    Useful for testing the pipeline flow without heavy computation.
    """

    def generate(
        self, n_structures: int, cycle: int = 0, metrics: dict[str, Any] | None = None
    ) -> Iterator[Structure]:
        """
        Generate mock structures.

        Args:
            n_structures: The number of structures to generate.
            cycle: The current active learning cycle number.
            metrics: Optional metrics from the previous cycle.

        Yields:
            Structure: Generated mock structure.
        """
        logger.info(f"Generating {n_structures} mock structures (Cycle {cycle})")

        if n_structures <= 0:
            logger.warning("n_structures must be positive")
            return

        cfg = self.config
        if not isinstance(cfg, MockGeneratorConfig):
            msg = f"Invalid config type for MockGenerator: {type(cfg)}"
            raise TypeError(msg)

        cell_size = cfg.cell_size
        n_atoms = cfg.n_atoms
        atomic_numbers = cfg.atomic_numbers

        # Ensure strict iterator behavior (no intermediate list)
        for _ in range(n_structures):
            pos = np.random.rand(n_atoms, 3) * cell_size
            # Cycle through atomic numbers if fewer than n_atoms
            full_numbers = np.resize(np.array(atomic_numbers), n_atoms)

            cell = np.eye(3) * cell_size
            pbc = np.array([True, True, True])
            yield Structure(positions=pos, atomic_numbers=full_numbers, cell=cell, pbc=pbc)

    def enhance(self, structure: Structure) -> Iterator[Structure]:
        """
        Mock enhancement: yield sufficient candidates to satisfy selection limit.
        """
        # Yield anchor
        yield structure

        # Yield dummy candidates to simulate local exploration
        # Orchestrator usually selects 6. So we need enough.
        for i in range(10):
            cand = structure.model_deep_copy()
            cand.tags["provenance"] = f"local_candidate_mock_{i}"
            # Perturb slightly to ensure uniqueness if checked
            cand.positions += np.random.uniform(-0.1, 0.1, size=cand.positions.shape)
            yield cand

    def __repr__(self) -> str:
        return f"<MockGenerator(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"MockGenerator({self.name})"
