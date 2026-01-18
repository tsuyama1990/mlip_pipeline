import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.orchestration.manager import WorkflowManager
from mlip_autopipec.orchestration.models import OrchestratorConfig, WorkflowState


@pytest.fixture
def mock_config(tmp_path: Path) -> MagicMock:
    config = MagicMock(spec=SystemConfig)
    config.working_dir = tmp_path
    config.db_path = tmp_path / "test.db"
    return config

@pytest.fixture
def mock_orch_config() -> OrchestratorConfig:
    return OrchestratorConfig(max_generations=2, workers=1, dask_scheduler_address=None)

@pytest.fixture
def manager(mock_config: MagicMock, mock_orch_config: OrchestratorConfig) -> WorkflowManager:
    with patch('mlip_autopipec.orchestration.manager.DatabaseManager'), \
         patch('mlip_autopipec.orchestration.manager.TaskQueue'), \
         patch('mlip_autopipec.orchestration.manager.Dashboard'):
        return WorkflowManager(mock_config, mock_orch_config)

def test_manager_init(manager: WorkflowManager) -> None:
    assert manager.state.current_generation == 0
    assert manager.state.status == "idle"

def test_load_state_existing(mock_config: MagicMock, mock_orch_config: OrchestratorConfig) -> None:
    state_file = mock_config.working_dir / "workflow_state.json"
    saved_state = WorkflowState(current_generation=1, status="training")
    state_file.write_text(saved_state.model_dump_json())

    with patch('mlip_autopipec.orchestration.manager.DatabaseManager'), \
         patch('mlip_autopipec.orchestration.manager.TaskQueue'), \
         patch('mlip_autopipec.orchestration.manager.Dashboard'):
        mgr = WorkflowManager(mock_config, mock_orch_config)
        assert mgr.state.current_generation == 1
        assert mgr.state.status == "training"

def test_save_state(manager: WorkflowManager) -> None:
    manager.state.current_generation = 1
    manager._save_state()

    state_file = manager.work_dir / "workflow_state.json"
    assert state_file.exists()
    loaded = json.loads(state_file.read_text())
    assert loaded['current_generation'] == 1

def test_run_loop_transitions(manager: WorkflowManager) -> None:
    # We mock the phases to just verify the loop logic
    manager._run_exploration_phase = MagicMock(side_effect=lambda: setattr(manager.state, 'status', 'dft'))  # type: ignore
    manager._run_dft_phase = MagicMock(side_effect=lambda: setattr(manager.state, 'status', 'training'))  # type: ignore
    manager._run_training_phase = MagicMock(side_effect=lambda: setattr(manager.state, 'status', 'inference'))  # type: ignore

    def inference_side_effect() -> None:
        manager.state.status = 'idle'
        manager.state.current_generation += 1

    manager._run_inference_phase = MagicMock(side_effect=inference_side_effect)  # type: ignore

    manager.run()

    assert manager.state.current_generation == 2
    assert manager.task_queue.shutdown.called  # type: ignore

def test_resilience_bad_state_file(mock_config: MagicMock, mock_orch_config: OrchestratorConfig) -> None:
    state_file = mock_config.working_dir / "workflow_state.json"
    state_file.write_text("{invalid_json")

    with patch('mlip_autopipec.orchestration.manager.DatabaseManager'), \
         patch('mlip_autopipec.orchestration.manager.TaskQueue'), \
         patch('mlip_autopipec.orchestration.manager.Dashboard'):
        mgr = WorkflowManager(mock_config, mock_orch_config)
        # Should fall back to defaults
        assert mgr.state.current_generation == 0
