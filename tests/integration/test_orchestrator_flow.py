import json
from pathlib import Path

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models import GlobalConfig, OrchestratorConfig


def test_orchestrator_flow(tmp_path: Path) -> None:
    config = GlobalConfig(
        orchestrator=OrchestratorConfig(work_dir=tmp_path / "test_run", max_cycles=2)
    )

    orchestrator = Orchestrator(config)
    orchestrator.run()

    state_file = tmp_path / "test_run/workflow_state.json"
    assert state_file.exists()

    with state_file.open("r") as f:
        data = json.load(f)

    assert data["iteration"] == 2
    assert data["status"] == "COMPLETED"

    # Check if log file exists
    log_file = tmp_path / "test_run/mlip.log"
    assert log_file.exists()
