"""Oracle (DFT) module implementation."""

import random

from pyacemaker.core.base import ModuleResult
from pyacemaker.core.exceptions import PYACEMAKERError
from pyacemaker.core.interfaces import Oracle
from pyacemaker.domain_models.models import StructureMetadata, StructureStatus


class MockOracle(Oracle):
    """Mock Oracle implementation for testing."""

    def run(self) -> ModuleResult:
        """Run the oracle (batch processing)."""
        self.logger.info("Running MockOracle")

        # Simulate failure based on config if needed
        if self.config.oracle.dft.parameters.get("simulate_failure", False):
            msg = "Simulated Oracle failure"
            raise PYACEMAKERError(msg)

        return ModuleResult(status="success")

    def compute_batch(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        """Compute energy/forces for a batch."""
        self.logger.info(f"Computing batch of {len(structures)} structures (mock)")

        computed = []
        for s in structures:
            # Update structure status
            s.status = StructureStatus.CALCULATED
            # Mock results with slight randomness
            energy = -100.0 + random.uniform(-1.0, 1.0)  # noqa: S311
            forces = [[random.uniform(-0.1, 0.1) for _ in range(3)]]  # noqa: S311

            s.features["energy"] = energy
            s.features["forces"] = forces
            computed.append(s)

        return computed
