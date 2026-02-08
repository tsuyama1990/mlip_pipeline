import logging
from collections.abc import Iterable, Iterator

import numpy as np

from mlip_autopipec.components.oracle.base import BaseOracle
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class MockOracle(BaseOracle):
    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        logger.info("Computing labels for structures")
        for s in structures:
            n_atoms = len(s.positions)

            # Generate random forces, but ensure sum is zero (Newton's 3rd law/translation invariance)
            raw_forces = np.random.rand(n_atoms, 3) - 0.5
            force_sum = np.sum(raw_forces, axis=0)
            correction = force_sum / n_atoms
            s.forces = raw_forces - correction

            s.energy = float(np.random.rand() * n_atoms * -3.0)
            s.stress = np.random.rand(6)
            yield s
