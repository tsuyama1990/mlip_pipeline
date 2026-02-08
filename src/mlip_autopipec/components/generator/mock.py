import logging
from collections.abc import Iterator

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

    def generate(self, n_structures: int) -> Iterator[Structure]:
        """
        Generate mock structures.

        Args:
            n_structures: The number of structures to generate.

        Yields:
            Structure: Generated mock structure.
        """
        logger.info(f"Generating {n_structures} mock structures")

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
            # But specific implies we want exactly these atoms?
            # MockGeneratorConfig says atomic_numbers is list[int].
            # Let's assume we tile them to match n_atoms or just use them if len matches.
            # If n_atoms > len(atomic_numbers), we repeat.

            # Simple approach: repeat
            full_numbers = np.resize(np.array(atomic_numbers), n_atoms)

            cell = np.eye(3) * cell_size
            pbc = np.array([True, True, True])
            yield Structure(positions=pos, atomic_numbers=full_numbers, cell=cell, pbc=pbc)
