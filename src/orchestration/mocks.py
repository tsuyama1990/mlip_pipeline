import logging
from pathlib import Path

from ase import Atoms

from src.config.config_model import GlobalConfig
from src.domain_models import Dataset, StructureMetadata, ValidationResult
from src.interfaces.core_interfaces import Explorer, Oracle, Trainer, Validator

logger = logging.getLogger(__name__)


class MockExplorer(Explorer):
    def generate(self, config: GlobalConfig) -> list[StructureMetadata]:
        """
        Mock generation of structures. Returns a single H2 molecule.
        """
        logger.info("MockExplorer generated structures")
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        return [StructureMetadata(structure=atoms, source="mock", generation_method="random")]


class MockOracle(Oracle):
    def calculate(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        """
        Mock calculation. Returns the structures as-is (labeled).
        """
        logger.info(f"MockOracle calculated {len(structures)} structures")
        return structures


class MockTrainer(Trainer):
    def train(self, dataset: Dataset, previous_potential: Path | None) -> Path:
        """
        Mock training. Returns a dummy potential path.
        """
        logger.info(f"MockTrainer updated potential using {len(dataset)} structures")
        return Path("mock_potential.yace")


class MockValidator(Validator):
    def validate(self, potential_path: Path) -> ValidationResult:
        """
        Mock validation. Always passes with RMSE=0.0.
        """
        logger.info(f"MockValidator validated potential at {potential_path}")
        return ValidationResult(passed=True, metrics={"rmse": 0.0})
