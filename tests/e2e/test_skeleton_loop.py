import logging
from pathlib import Path

import pytest

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.orchestration.orchestrator import Orchestrator


def test_skeleton_loop(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    caplog.set_level(logging.INFO)

    config = GlobalConfig(work_dir=tmp_path / "workspace", max_cycles=3, random_seed=42)

    orchestrator = Orchestrator(config)
    orchestrator.run_loop()

    assert "Starting Active Learning Cycle" in caplog.text
    # We expect 3 cycles
    assert caplog.text.count("MockExplorer generated structures") == 3
    assert caplog.text.count("MockTrainer updated potential") == 3
    assert "Cycle 3/3 completed" in caplog.text
