import logging
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

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
        # rng argument removed from __init__ to simplify signature matching with Factory.
        # If needed for testing, use config.seed
        super().__init__(config)
        self.config: MockDynamicsConfig = config

        # Initialize RNG
        # Use config.seed if provided, else random
        self._rng = random.Random(config.seed)  # noqa: S311

    def explore(
        self,
        potential: Potential,
        start_structures: Iterable[Structure],
        workdir: Path | None = None,
        physics_baseline: dict[str, Any] | None = None,
    ) -> Iterator[Structure]:
        """
        Explore the potential energy surface starting from given structures.

        Args:
            potential: The current potential (ignored in Mock).
            start_structures: An iterable of starting structures.
            workdir: Directory to write exploration files (ignored in Mock).
            physics_baseline: Optional physics baseline configuration (ignored in Mock).

        Yields:
            Structure: Selected structures with simulated uncertainty.
        """
        logger.info("Exploring structures for uncertainty")
        count = 0
        selection_rate = self.config.selection_rate
        # Add simulated uncertainty to cross the threshold
        simulated_uncertainty = (
            self.config.uncertainty_threshold + self.config.simulated_uncertainty
        )

        for s in start_structures:
            # Randomly select structures to simulate finding "uncertain" regions
            if self._rng.random() < selection_rate:
                # Create a deep copy using our custom method
                s_copy = s.model_deep_copy()
                s_copy.uncertainty = simulated_uncertainty
                yield s_copy
                count += 1

        logger.info(f"Found {count} uncertain structures")

    def __repr__(self) -> str:
        return f"<MockDynamics(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"MockDynamics({self.name})"
