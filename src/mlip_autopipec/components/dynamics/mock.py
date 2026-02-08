import logging
import secrets
from collections.abc import Iterable, Iterator

from mlip_autopipec.components.dynamics.base import BaseDynamics
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class MockDynamics(BaseDynamics):
    def explore(
        self, potential: Potential, start_structures: Iterable[Structure]
    ) -> Iterator[Structure]:
        logger.info("Exploring structures for uncertainty")
        count = 0
        selection_rate = self.config.selection_rate

        # In Cycle 01, start_structures might be just generated ones.
        # Randomly select subset
        for s in start_structures:
            # secrets.randbelow(100) returns [0, 99], /100.0 -> [0.0, 0.99]
            if (secrets.randbelow(100) / 100.0) < selection_rate:
                yield s
                count += 1
        logger.info(f"Found {count} uncertain structures")
