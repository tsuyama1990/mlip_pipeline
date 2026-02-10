import pytest
from pathlib import Path
from mlip_autopipec.domain_models import Config
from mlip_autopipec.core.orchestrator import Orchestrator

def test_orchestrator_mock_loop(tmp_path: Path) -> None:
    """
    UAT-01: Basic Pipeline Execution (Mock Mode)
    """
    # 1. Setup Config
    config_data = {
        "orchestrator": {
            "work_dir": str(tmp_path / "work"),
            "max_iterations": 2
        },
        "generator": {
            "type": "mock",
            "params": {}
        },
        "oracle": {
            "type": "mock"
        },
        "trainer": {
            "type": "mock",
            "dataset_path": str(tmp_path / "data.xyz")
        }
    }
    config = Config.model_validate(config_data)

    # 2. Init Orchestrator
    orchestrator = Orchestrator(config)

    # 3. Run
    orchestrator.run_loop()

    # 4. Verify State
    state = orchestrator.state_manager.load()
    assert state.iteration == 2

    # Verify files
    work_dir = tmp_path / "work"
    assert (work_dir / "workflow_state.json").exists()
    assert (work_dir / "iteration_1" / "train.xyz").exists()
    assert (work_dir / "iteration_2" / "train.xyz").exists()
