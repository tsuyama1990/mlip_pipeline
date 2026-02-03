import logging
from pathlib import Path

from mlip_autopipec.domain_models.structures import CandidateStructure, StructureMetadata
from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult
from mlip_autopipec.orchestration.interfaces import Explorer, Oracle, Validator

logger = logging.getLogger(__name__)


class MockExplorer(Explorer):
    def explore(self, potential_path: Path | None, work_dir: Path) -> list[CandidateStructure]:
        logger.info("MockExplorer: exploring...")
        # Create a dummy structure file
        structure_file = work_dir / "candidate_0.xyz"
        structure_file.touch()
        return [
            CandidateStructure(
                structure_path=structure_file,
                metadata=StructureMetadata(source="mock_exploration", uncertainty=0.1),
            )
        ]


class MockOracle(Oracle):
    def compute(self, candidates: list[CandidateStructure], work_dir: Path) -> list[Path]:
        logger.info(f"MockOracle: computing for {len(candidates)} candidates...")
        results = []
        for i, _ in enumerate(candidates):
            result_file = work_dir / f"result_{i}.extxyz"
            result_file.touch()
            results.append(result_file)
        return results


class MockValidator(Validator):
    def validate(self, potential_path: Path, work_dir: Path) -> ValidationResult:
        logger.info("MockValidator: validating...")
        metric = MetricResult(name="mock_test", passed=True, score=0.001)
        return ValidationResult(passed=True, metrics=[metric])
