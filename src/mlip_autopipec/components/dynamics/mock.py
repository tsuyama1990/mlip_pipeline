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

    This mocks the complex "OTF Halt & Diagnose" loop described in the Spec (Section 3.4).
    Instead of running LAMMPS and halting on high uncertainty, it iterates over candidate
    structures and probabilistically selects them, tagging them with high uncertainty.
    """

    def __init__(self, config: MockDynamicsConfig) -> None:
        super().__init__(config)
        self.config: MockDynamicsConfig = config

        # Explicitly validate seed to prevent unsafe usage
        if config.seed is not None and not isinstance(config.seed, int):
            msg = f"Seed must be an integer or None, got {type(config.seed)}"
            raise TypeError(msg)

        self._rng = random.Random(config.seed)  # noqa: S311

    def explore(
        self, potential: Potential | None, start_structures: Iterable[Structure]
    ) -> Iterator[Structure]:
        """
        Explore the potential energy surface starting from given structures.

        Args:
            potential: The current potential (can be None for mock).
            start_structures: An iterable of starting structures.

        Yields:
            Structure: Selected structures with simulated uncertainty.
        """
        logger.info("Exploring structures for uncertainty")
        count = 0
        selection_rate = self.config.selection_rate
        uncertainty_threshold = self.config.uncertainty_threshold
        simulated_uncertainty = uncertainty_threshold + self.config.simulated_uncertainty

        for s in start_structures:
            # Randomly select structures to simulate finding "uncertain" regions
            if self._rng.random() < selection_rate:
                # Create a copy with updated uncertainty
                s_copy = s.model_copy(update={"uncertainty": simulated_uncertainty})
                yield s_copy
                count += 1

        logger.info(f"Found {count} uncertain structures")
