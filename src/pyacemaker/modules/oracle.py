"""Oracle (DFT) module implementation."""

from pyacemaker.core.base import ModuleResult
from pyacemaker.core.interfaces import Oracle
from pyacemaker.domain_models.models import StructureMetadata, StructureStatus


class MockOracle(Oracle):
    """Mock Oracle implementation for testing."""

    def run(self) -> ModuleResult:
        """Run the oracle (batch processing)."""
        self.logger.info("Running MockOracle")
        # In a real scenario, this would read a queue and process it
        return ModuleResult(status="success")

    def compute_batch(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        """Compute energy/forces for a batch."""
        self.logger.info(f"Computing batch of {len(structures)} structures (mock)")

        computed = []
        for s in structures:
            # Update structure status
            s.status = StructureStatus.CALCULATED
            # Mock results
            s.features["energy"] = -100.0  # Dummy value
            s.features["forces"] = [[0.0, 0.0, 0.0]]  # Dummy value
            computed.append(s)

        return computed
