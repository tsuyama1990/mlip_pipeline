import logging
from pathlib import Path

from ase import Atoms

from config import GlobalConfig
from domain_models import Dataset, StructureMetadata
from domain_models.dataset import ValidationResult

logger = logging.getLogger("mlip_pipeline.mocks")


class MockExplorer:
    def generate(self, config: GlobalConfig) -> list[StructureMetadata]:
        logger.info("MockExplorer generated structures")
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        return [
            StructureMetadata(structure=atoms, source="mock_explorer", generation_method="dummy")
        ]


class MockOracle:
    def calculate(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        logger.info(f"MockOracle calculated energies for {len(structures)} structures")
        return structures


class MockTrainer:
    def train(self, dataset: Dataset, previous_potential: Path | None) -> Path:
        logger.info("MockTrainer updated potential")
        return Path("mock_potential.pwo")


class MockValidator:
    def validate(self, potential_path: Path) -> ValidationResult:
        logger.info(f"MockValidator validated {potential_path}")
        return ValidationResult(metrics={"accuracy": 0.99}, passed=True)
