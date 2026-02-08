import logging
from collections.abc import Iterator

import numpy as np

from mlip_autopipec.domain_models import Potential, Structure
from mlip_autopipec.interfaces import BaseDynamics

logger = logging.getLogger(__name__)


class MockDynamics(BaseDynamics):
    """
    Mock implementation of a dynamics engine.
    Returns dummy structures representing sampled configurations.
    """

    def run(self, potential: Potential) -> Iterator[Structure]:
        """
        Simulates dynamics by generating a few dummy structures.
        """
        logger.info(f"MockDynamics: Running MD with potential {potential.version}...")

        # Generate 3 dummy structures
        for i in range(3):
            structure = Structure(
                positions=np.array([[float(i), 0.0, 0.0]]),
                atomic_numbers=np.array([1]),
                cell=np.eye(3),
                pbc=(True, True, True),
                properties={"md_step": i, "uncertainty": 0.1 * (i + 1)},
            )
            yield structure
