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
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir

    def generate_candidates(self, config: ExplorationConfig) -> list[StructureMetadata]:
        logger.info(f"MockExplorer generated candidates using strategy: {config.strategy_name}")
        candidates = []
        for _ in range(config.max_structures):
            atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
            struct_id = str(uuid.uuid4())

            # Optimized: No disk write in inner loop.
            # Only create metadata in memory.
            meta = StructureMetadata(
                id=struct_id,
                source="mock_explorer",
                generation_method=config.strategy_name,
                structure=atoms,
                filepath=None,  # Not saving to disk
            )
            candidates.append(meta)
        return candidates


class MockOracle:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir

    def calculate(
        self, structures: list[StructureMetadata], config: DFTConfig
    ) -> list[StructureMetadata]:
        logger.info(
            f"MockOracle calculated properties for {len(structures)} structures with {config.calculator}"
        )
        if not structures:
            logger.warning("MockOracle received empty structure list")
            return []

        labeled_structures = []
        for s in structures:
            # Scalability: Clone structure to avoid in-place modification bloat/side-effects
            new_atoms = s.structure.copy()  # type: ignore[no-untyped-call]

            # Simulate calculation
            new_atoms.info["energy"] = -10.0
            new_atoms.arrays["forces"] = np.zeros((len(new_atoms), 3))

            # Create new metadata object
            new_meta = s.model_copy(update={"structure": new_atoms})
            new_meta.selection_score = 0.5  # Dummy

            labeled_structures.append(new_meta)

        return labeled_structures


class MockTrainer:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir

    def train(self, structures: list[StructureMetadata], config: TrainingConfig) -> str:
        logger.info(
            f"MockTrainer updated potential using {len(structures)} structures. Fitting code: {config.fitting_code}"
        )
        if not structures:
            logger.warning("MockTrainer received empty structure list")
            # Return existing potential or dummy if none
            return str(self.work_dir / "mock_potential.yace")

        # Create a dummy potential file in work_dir
        potential_path = self.work_dir / "mock_potential.yace"
        potential_path.write_text("mock potential content")
        return str(potential_path)


class MockValidator:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir

    def validate(self, potential_path: str) -> ValidationResult:
        logger.info(f"MockValidator validating potential: {potential_path}")
        return ValidationResult(
            passed=True,
            metrics=[MetricResult(name="test_metric", passed=True, score=0.99)],
            report_path=str(self.work_dir / "validation_report.json"),
        )
