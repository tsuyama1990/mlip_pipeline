import logging
import secrets
from typing import Iterator, Optional, Any

import numpy as np

from mlip_autopipec.domain_models import Potential, Structure
from mlip_autopipec.interfaces import BaseGenerator

logger = logging.getLogger(__name__)


class MockGenerator(BaseGenerator):
    """
    Mock implementation of a structure generator.
    Generates dummy structures for testing the pipeline.
    """

    def __init__(self, fail_rate: float = 0.0, empty_rate: float = 0.0, **kwargs: Any) -> None:
        """
        Args:
            fail_rate: Probability of failure during generation (0.0 to 1.0).
            empty_rate: Probability of returning empty iterator (0.0 to 1.0).
            **kwargs: Ignored extra arguments.
        """
        self.fail_rate = fail_rate
        self.empty_rate = empty_rate
        self.rng = np.random.default_rng(secrets.randbits(128))
        if kwargs:
            logger.debug(f"MockGenerator received extra args: {kwargs}")

    def generate(self, potential: Optional[Potential] = None) -> Iterator[Structure]:
        """
        Generates 5 dummy structures.
        """
        if self.rng.random() < self.fail_rate:
            msg = "MockGenerator: Simulated failure during generation."
            raise RuntimeError(msg)

        if self.rng.random() < self.empty_rate:
            logger.info("MockGenerator: Returning empty iterator (simulated).")
            return

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
