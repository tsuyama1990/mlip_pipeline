import logging
from pathlib import Path

from ase import Atoms

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.domain_models import Dataset, StructureMetadata, ValidationResult

logger = logging.getLogger(__name__)


class MockExplorer:
    def generate(self, config: GlobalConfig) -> list[StructureMetadata]:
        logger.info("MockExplorer generated structures")
        # Return a dummy structure
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        return [
            StructureMetadata(structure=atoms, source="MockExplorer", generation_method="dummy")
        ]


class MockOracle:
    def calculate(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        logger.info(f"MockOracle calculated energies for {len(structures)} structures")
        return structures


class MockTrainer:
    def train(self, dataset: Dataset, previous_potential: Path | None) -> Path:
        logger.info("MockTrainer updated potential")
        return Path("mock_potential.pth")


class MockValidator:
    def validate(self, potential_path: Path) -> ValidationResult:
        logger.info("MockValidator validated potential")
        return ValidationResult(metrics={"rmse": 0.0}, passed=True, details={"msg": "All good"})
