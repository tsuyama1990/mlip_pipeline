import logging
from collections.abc import Iterable, Iterator

import numpy as np

from mlip_autopipec.domain_models import Structure
from mlip_autopipec.interfaces import BaseOracle

logger = logging.getLogger(__name__)


class MockOracle(BaseOracle):
    """
    Mock implementation of an oracle (calculator).
    Adds dummy energy, forces, and stress to structures.
    """

    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """
        Computes dummy labels for input structures.
        """
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
