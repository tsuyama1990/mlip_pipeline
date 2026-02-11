from pathlib import Path

import pytest

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models import GlobalConfig, OrchestratorConfig


def test_orchestrator_init_dir_error(tmp_path: Path) -> None:
    work_dir = tmp_path / "file_exists"
    work_dir.touch()

    config = GlobalConfig(
        orchestrator=OrchestratorConfig(work_dir=work_dir, max_cycles=1)
    )

    with pytest.raises(NotADirectoryError):
        Orchestrator(config)


def test_orchestrator_resume_log(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    work_dir = tmp_path / "resume_test"
    work_dir.mkdir()

    state_file = work_dir / "workflow_state.json"
    state_file.write_text('{"iteration": 2, "status": "EXPLORATION"}')

    config = GlobalConfig(
        orchestrator=OrchestratorConfig(work_dir=work_dir, max_cycles=5)
    )

    with caplog.at_level("INFO"):
        Orchestrator(config)

    assert "Resuming from iteration 2" in caplog.text
