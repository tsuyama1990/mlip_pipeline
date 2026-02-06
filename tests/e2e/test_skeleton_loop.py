import logging
from pathlib import Path

import pytest

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.orchestration.mocks import (
    MockExplorer,
    MockOracle,
    MockTrainer,
    MockValidator,
)
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.utils.logging import setup_logging


@pytest.fixture
def configure_logging() -> None:
    setup_logging(level=logging.INFO)


def test_mock_loop_execution(
    configure_logging: None, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    # Setup config
    config = GlobalConfig(work_dir=tmp_path / "work", max_cycles=2, random_seed=42)

    # Instantiate mocks
    explorer = MockExplorer()
    oracle = MockOracle()
    trainer = MockTrainer()
    validator = MockValidator()

    # Instantiate orchestrator
    orchestrator = Orchestrator(
        config=config,
        explorer=explorer,
        oracle=oracle,
        trainer=trainer,
        validator=validator,
    )

    # Run loop
    with caplog.at_level(logging.INFO):
        orchestrator.run_loop()

    # Verify logs
    assert "Starting pipeline for 2 cycles" in caplog.text
    assert "Cycle 1/2" in caplog.text
    assert "Cycle 2/2" in caplog.text
    # MockExplorer generates 1 structure per cycle.
    # Cycle 1: 1 structure added. Trainer sees 1.
    # Cycle 2: 1 structure added. Trainer sees 2.
    assert "MockTrainer training on 1 structures" in caplog.text
    assert "MockTrainer training on 2 structures" in caplog.text
    assert "Pipeline completed" in caplog.text
