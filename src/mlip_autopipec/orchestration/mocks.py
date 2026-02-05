import logging
import random
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.domain_models.dataset import Dataset
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult

logger = logging.getLogger(__name__)

class MockExplorer:
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.rng = random.Random(config.random_seed) # noqa: S311

    def generate_candidates(self) -> list[StructureMetadata]:
        logger.info("MockExplorer generating structures...")
        count = self.config.exploration.max_structures
        candidates = []
        for _ in range(count):
            # Generate random structure
            atoms = Atoms('Cu2', positions=[[0, 0, 0], [1.5, 0, 0]])
            # rattle might be untyped in some versions
            atoms.rattle(stdev=0.1, seed=self.rng.randint(0, 10000)) # type: ignore
            meta = StructureMetadata(
                structure=atoms,
                source="mock_explorer",
                generation_method="random_displacement"
            )
            candidates.append(meta)
        return candidates

class MockOracle:
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        # Use numpy rng for vector operations if needed, seeded from config
        self.rng = np.random.default_rng(config.random_seed)

    def calculate(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        logger.info(f"MockOracle calculating {len(structures)} structures...")

        # Simulate batch processing
        # In a real implementation, we would write all structures to a file, run DFT, and read back.
        # Here we loop, but the interface supports passing the whole list.

        for meta in structures:
            atoms = meta.structure
            n_atoms = len(atoms)
            # Deterministic random values based on seed
            energy = -3.5 * n_atoms + self.rng.normal(0, 0.1)
            forces = self.rng.normal(0, 0.1, (n_atoms, 3))

            calc = SinglePointCalculator(atoms, energy=energy, forces=forces) # type: ignore
            atoms.calc = calc

        return structures

class MockTrainer:
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config

    def train(self, dataset: Dataset) -> Path:
        logger.info(f"MockTrainer training on dataset with {len(dataset)} structures...")

        output_path = self.config.work_dir / "potential.yace"
        # Use Path.open()
        with output_path.open("w") as f:
            f.write("MOCK POTENTIAL CONTENT")

        logger.info(f"MockTrainer updated potential at {output_path}")
        return output_path

class MockValidator:
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config

    def validate(self, potential: Path) -> ValidationResult:
        logger.info("MockValidator verifying potential...")

        report_path = self.config.work_dir / "validation_report.pdf"

        # Return a passing result for Cycle 01
        return ValidationResult(
            passed=True,
            metrics=[
                MetricResult(name="Elastic Tensor", passed=True, score=0.01),
                MetricResult(name="Phonons", passed=True, score=0.02)
            ],
            report_path=str(report_path)
        )
