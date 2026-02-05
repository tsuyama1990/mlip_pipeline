import logging
import uuid
from pathlib import Path

import numpy as np
from ase import Atoms

from mlip_autopipec.config.config_model import DFTConfig, ExplorationConfig, TrainingConfig
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult

logger = logging.getLogger(__name__)


class MockExplorer:
    def generate_candidates(self, config: ExplorationConfig) -> list[StructureMetadata]:
        logger.info(f"MockExplorer generated candidates using strategy: {config.strategy_name}")
        candidates = []
        for _ in range(config.max_structures):
            atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
            struct_id = str(uuid.uuid4())

            # Save to disk
            filename = f"mock_structure_{struct_id}.xyz"
            atoms.write(filename)  # type: ignore[no-untyped-call]

            meta = StructureMetadata(
                id=struct_id,
                source="mock_explorer",
                generation_method=config.strategy_name,
                structure=atoms,
                filepath=filename,
            )
            candidates.append(meta)
        return candidates


class MockOracle:
    def calculate(
        self, structures: list[StructureMetadata], config: DFTConfig
    ) -> list[StructureMetadata]:
        logger.info(
            f"MockOracle calculated properties for {len(structures)} structures with {config.calculator}"
        )
        for s in structures:
            # Simulate calculation
            s.structure.info["energy"] = -10.0
            # Add dummy forces (N_atoms, 3)
            s.structure.arrays["forces"] = np.zeros((len(s.structure), 3))
            s.selection_score = 0.5  # Dummy
        return structures


class MockTrainer:
    def train(self, structures: list[StructureMetadata], config: TrainingConfig) -> str:
        logger.info(
            f"MockTrainer updated potential using {len(structures)} structures. Fitting code: {config.fitting_code}"
        )
        # Create a dummy potential file
        potential_path = Path("mock_potential.yace")
        potential_path.write_text("mock potential content")
        return str(potential_path)


class MockValidator:
    def validate(self, potential_path: str) -> ValidationResult:
        logger.info(f"MockValidator validating potential: {potential_path}")
        return ValidationResult(
            passed=True,
            metrics=[MetricResult(name="test_metric", passed=True, score=0.99)],
            report_path="validation_report.json",
        )
