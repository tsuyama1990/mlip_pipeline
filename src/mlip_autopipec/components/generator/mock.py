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

        # Default parameters, overridable by component config or method config
        # Component config (self.config) is the baseline
        # Method config (config) overrides component config for this specific call

        effective_config = self.config.copy()
        if config:
            effective_config.update(config)

        cell_size = effective_config.get("cell_size", 10.0)
        n_atoms = effective_config.get("n_atoms", 2)
        atomic_numbers = effective_config.get("atomic_numbers", [1, 1])

        for _ in range(n_structures):
            pos = np.random.rand(n_atoms, 3) * cell_size
            numbers = np.array(atomic_numbers)
            cell = np.eye(3) * cell_size
            pbc = np.array([True, True, True])
            yield Structure(positions=pos, atomic_numbers=numbers, cell=cell, pbc=pbc)
