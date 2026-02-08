import logging
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np

from mlip_autopipec.components.generator.base import BaseGenerator
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class MockGenerator(BaseGenerator):
    DEFAULT_CELL_SIZE: ClassVar[float] = 10.0
    DEFAULT_N_ATOMS: ClassVar[int] = 2
    DEFAULT_ATOMIC_NUMBERS: ClassVar[list[int]] = [1, 1]

    def generate(
        self, n_structures: int, config: dict[str, Any] | None = None
    ) -> Iterator[Structure]:
        logger.info(f"Generating {n_structures} mock structures")

        # Merge configuration: method config overrides component config
        # Convert Pydantic model to dict to allow merging with override dict
        effective_config = self.config.model_dump()
        if config:
            effective_config.update(config)

        cell_size = float(effective_config.get("cell_size", self.DEFAULT_CELL_SIZE))
        n_atoms = int(effective_config.get("n_atoms", self.DEFAULT_N_ATOMS))
        atomic_numbers = effective_config.get("atomic_numbers", self.DEFAULT_ATOMIC_NUMBERS)

        # Ensure strict iterator behavior (no intermediate list)
        for _ in range(n_structures):
            pos = np.random.rand(n_atoms, 3) * cell_size
            numbers = np.array(atomic_numbers)
            cell = np.eye(3) * cell_size
            pbc = np.array([True, True, True])
            yield Structure(positions=pos, atomic_numbers=numbers, cell=cell, pbc=pbc)
