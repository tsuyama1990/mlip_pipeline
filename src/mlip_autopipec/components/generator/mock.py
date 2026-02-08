import logging
from collections.abc import Iterator
from typing import Any

import numpy as np

from mlip_autopipec.components.generator.base import BaseGenerator
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class MockGenerator(BaseGenerator):
    def generate(
        self, n_structures: int, config: dict[str, Any] | None = None
    ) -> Iterator[Structure]:
        logger.info(f"Generating {n_structures} mock structures")

        # Merge configuration: method config overrides component config
        effective_config = self.config.model_dump()
        if config:
            effective_config.update(config)

        cell_size = float(effective_config["cell_size"])
        n_atoms = int(effective_config["n_atoms"])
        atomic_numbers = effective_config["atomic_numbers"]

        # Ensure strict iterator behavior (no intermediate list)
        for _ in range(n_structures):
            pos = np.random.rand(n_atoms, 3) * cell_size
            numbers = np.array(atomic_numbers)
            cell = np.eye(3) * cell_size
            pbc = np.array([True, True, True])
            yield Structure(positions=pos, atomic_numbers=numbers, cell=cell, pbc=pbc)
