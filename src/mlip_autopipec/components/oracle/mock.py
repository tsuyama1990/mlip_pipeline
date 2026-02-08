import logging
import secrets
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from mlip_autopipec.domain_models import Structure
from mlip_autopipec.interfaces import BaseOracle

logger = logging.getLogger(__name__)


class MockOracle(BaseOracle):
    """
    Mock implementation of an oracle (calculator).
    Adds dummy energy, forces, and stress to structures.
    """

    def __init__(self, fail_rate: float = 0.0, **kwargs: Any) -> None:
        """
        Args:
            fail_rate: Probability of failure during computation (0.0 to 1.0).
            **kwargs: Ignored extra arguments.
        """
        self.fail_rate = fail_rate
        self.rng = np.random.default_rng(secrets.randbits(128))
        if kwargs:
            logger.debug(f"MockOracle received extra args: {kwargs}")

    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """
        Computes dummy labels for input structures.
        """
        if self.rng.random() < self.fail_rate:
            msg = "MockOracle: Simulated failure during computation."
            raise RuntimeError(msg)

        logger.info("MockOracle: Computing labels for structures...")

        for i, structure in enumerate(structures):
            n_atoms = len(structure.atomic_numbers)

            # Assign dummy labels
            # Energy: -1.0 per atom
            structure.energy = -1.0 * n_atoms

            # Forces: Zero forces
            structure.forces = np.zeros((n_atoms, 3))

            # Stress: Zero stress
            structure.stress = np.zeros((3, 3))

            # Add metadata
            if structure.properties is None:
                structure.properties = {}
            structure.properties["oracle_computed"] = True
            structure.properties["mock_compute_id"] = i

            yield structure
