import logging
import math
import random
from collections.abc import Iterable, Iterator

from mlip_autopipec.components.dynamics.base import BaseDynamics
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class MockDynamics(BaseDynamics):
    """
    Mock implementation of the Dynamics component.

    Simulates exploration by randomly selecting structures based on a configured
    selection rate and assigning a simulated uncertainty value.
    """

    def explore(
        self, potential: Potential, start_structures: Iterable[Structure]
    ) -> Iterator[Structure]:
        """
        Explore the potential energy surface starting from given structures.

        Iterates through start_structures and selects a subset based on `selection_rate`.
        Selected structures are assigned an `uncertainty` property simulating
        high uncertainty (above threshold).

        Args:
            potential: The current potential (not used in simulation logic but required by interface).
            start_structures: An iterable of starting structures.

        Yields:
            Structure: Selected structures with simulated uncertainty.
        """
        logger.info("Exploring structures for uncertainty")
        count = 0
        selection_rate = self.config.selection_rate
        # Use config for threshold, defaulting to reasonable value if somehow missing (though Pydantic enforces it)
        # Note: In DynamicsConfig, uncertainty_threshold has default 5.0
        uncertainty_threshold = self.config.uncertainty_threshold

        # In Cycle 01, start_structures might be just generated ones.
        # Ensure we stream by iterating over the input iterable
        # Optimize using geometric skipping to minimize RNG calls
        iterator = iter(start_structures)

        while True:
            # Determine how many items to skip
            # If selection_rate is 1.0, we take everything (skip=0)
            if selection_rate >= 1.0:
                skip = 0
            elif selection_rate <= 0.0:
                break
            else:
                # Geometric distribution: number of failures before first success
                # We use random.random() which is [0.0, 1.0). If it's 0.0, log is -inf, need to handle.
                r = random.random()  # noqa: S311
                if r == 0.0:
                    r = 1e-10  # Avoid log(0)
                skip = int(math.log(r) / math.log(1 - selection_rate))

            # Consume 'skip' items
            try:
                for _ in range(skip):
                    next(iterator)

                # Take the next one
                s = next(iterator)
                s.tags = {"uncertainty": uncertainty_threshold + 1.0}
                yield s
                count += 1

            except StopIteration:
                break

        logger.info(f"Found {count} uncertain structures")
