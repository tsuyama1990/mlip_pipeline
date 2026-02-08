import logging
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
        # Use simple random sampling as requested for mock optimization
        for s in start_structures:
            # We use random.random() which is acceptable for mock simulation
            if random.random() < selection_rate:  # noqa: S311
                s.tags = {"uncertainty": uncertainty_threshold + 1.0}
                yield s
                count += 1

        logger.info(f"Found {count} uncertain structures")
