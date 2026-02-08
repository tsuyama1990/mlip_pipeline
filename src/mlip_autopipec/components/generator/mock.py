import logging
from collections.abc import Iterator

import numpy as np

from mlip_autopipec.domain_models import Potential, Structure
from mlip_autopipec.interfaces import BaseGenerator

logger = logging.getLogger(__name__)


class MockGenerator(BaseGenerator):
    """
    Mock implementation of a structure generator.
    Generates dummy structures for testing the pipeline.
    """

    def generate(self, potential: Potential | None = None) -> Iterator[Structure]:
        """
        Generates 5 dummy structures.
        """
        logger.info("MockGenerator: Generating 5 dummy structures...")

        for i in range(5):
            # Create a dummy structure (single H atom)
            structure = Structure(
                positions=np.array([[0.0, 0.0, 0.0]]),
                atomic_numbers=np.array([1]),
                cell=np.eye(3),
                pbc=(True, True, True),
                properties={"mock_id": i, "source": "MockGenerator"},
            )
            yield structure
