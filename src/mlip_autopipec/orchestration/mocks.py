import logging
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.potential import Potential

logger = logging.getLogger(__name__)


class MockExplorer:
    """Mock implementation of the Explorer protocol."""

    def explore(self, potential: Potential | None, work_dir: Path) -> dict[str, Any]:
        logger.info(f"MockExplorer exploring in {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)

        # Simulate finding candidate structures
        candidates_dir = work_dir / "candidates"
        candidates_dir.mkdir(exist_ok=True)
        candidate_file = candidates_dir / "mock_candidate.xyz"
        candidate_file.touch()

        return {
            "halted": True,
            "candidates": [candidate_file],
            "reason": "Mock uncertainty threshold exceeded"
        }


class MockOracle:
    """Mock implementation of the Oracle protocol."""

    def compute(self, structures: list[Path], work_dir: Path) -> list[Path]:
        logger.info(f"MockOracle computing {len(structures)} structures in {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, _struct in enumerate(structures):
            result_file = work_dir / f"result_{i}.pckl"
            result_file.touch()
            results.append(result_file)

        return results


class MockValidator:
    """Mock implementation of the Validator protocol."""

    def validate(self, potential: Path) -> dict[str, Any]:
        logger.info(f"MockValidator validating {potential}")
        return {
            "passed": True,
            "metrics": {"rmse_energy": 0.001, "rmse_force": 0.01}
        }
