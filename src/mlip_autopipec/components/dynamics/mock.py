import logging
import random
from collections.abc import Iterable, Iterator

from mlip_autopipec.components.dynamics.base import BaseDynamics
from mlip_autopipec.domain_models.config import MockDynamicsConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class MockDynamics(BaseDynamics):
    """
    Mock implementation of the Dynamics component.

    Simulates exploration by randomly selecting structures based on a configured
    selection rate and assigning a simulated uncertainty value.
    """

    def __init__(self, config: MockDynamicsConfig) -> None:
        super().__init__(config)
        self.config: MockDynamicsConfig = config
        self._rng = random.Random(config.seed)  # noqa: S311

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
        uncertainty_threshold = self.config.uncertainty_threshold
        simulated_uncertainty = uncertainty_threshold + self.config.simulated_uncertainty

        # In Cycle 01, start_structures might be just generated ones.
        # Ensure we stream by iterating over the input iterable
        # Use simple random sampling as requested for mock optimization
        for s in start_structures:
            # Use local RNG for reproducibility if seed is provided
            if self._rng.random() < selection_rate:
                # Create a copy with updated uncertainty (immutability)
                s_copy = s.model_copy(update={"uncertainty": simulated_uncertainty})
                yield s_copy
                count += 1

        logger.info(f"Found {count} uncertain structures")
