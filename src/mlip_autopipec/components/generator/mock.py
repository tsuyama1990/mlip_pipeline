import logging
from typing import Any

import numpy as np

from mlip_autopipec.components.generator.base import BaseGenerator
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class MockGenerator(BaseGenerator):
    def generate(self, n_structures: int, config: dict[str, Any] | None = None) -> list[Structure]:
        logger.info(f"Generating {n_structures} mock structures")
        structures: list[Structure] = []
        for _ in range(n_structures):
            pos = np.random.rand(2, 3) * 10
            numbers = np.array([1, 1])
            cell = np.eye(3) * 10
            pbc = np.array([True, True, True])
            structures.append(Structure(positions=pos, atomic_numbers=numbers, cell=cell, pbc=pbc))
        return structures
