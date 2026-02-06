import logging
from pathlib import Path

from ase import Atoms

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.domain_models.dataset import Dataset
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.domain_models.validation import ValidationResult

logger = logging.getLogger(__name__)


class MockExplorer:
    """Mock implementation of the Explorer interface."""

    def generate(self, config: GlobalConfig) -> list[StructureMetadata]:
        """Generate dummy structures."""
        logger.info("MockExplorer generating structures...")
        atoms = Atoms("Cu", positions=[[0, 0, 0]])
        return [
            StructureMetadata(structure=atoms, source="active_learning", generation_method="mock")
        ]


class MockOracle:
    """Mock implementation of the Oracle interface."""

    def calculate(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        """Simulate calculation."""
        logger.info(f"MockOracle calculating {len(structures)} structures...")
        return structures


class MockTrainer:
    """Mock implementation of the Trainer interface."""

    def train(self, dataset: Dataset, previous_potential: Path | None) -> Path:
        """Simulate training."""
        logger.info(f"MockTrainer training on {len(dataset)} structures...")
        return Path("mock_potential.pt")


class MockValidator:
    """Mock implementation of the Validator interface."""

    def validate(self, potential_path: Path) -> ValidationResult:
        """Simulate validation."""
        logger.info(f"MockValidator validating {potential_path}...")
        return ValidationResult(metrics={"rmse": 0.0}, is_acceptable=True)
