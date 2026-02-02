import logging
from pathlib import Path
from typing import Any

from mlip_autopipec.orchestration.interfaces import Explorer, Oracle, Validator

logger = logging.getLogger(__name__)


class MockExplorer(Explorer):
    def explore(self, potential_path: Path | None, work_dir: Path) -> Any:
        logger.info(f"Phase: Exploration (Mock) in {work_dir}")
        return {"status": "mock_exploration_done"}


class MockOracle(Oracle):
    def compute(self, input_data: Any, work_dir: Path) -> Any:
        logger.info(f"Phase: Oracle (Mock) in {work_dir}")
        return {"status": "mock_oracle_done"}


class MockValidator(Validator):
    def validate(self, potential_path: Path) -> Any:
        logger.info(f"Phase: Validation (Mock) for {potential_path}")
        return {"status": "passed"}
