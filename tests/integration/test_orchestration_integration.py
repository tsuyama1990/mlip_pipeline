import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.orchestration.manager import WorkflowManager
from mlip_autopipec.orchestration.models import OrchestratorConfig


@pytest.fixture
def mock_config(tmp_path: Path) -> MagicMock:
    config = MagicMock(spec=SystemConfig)
    config.working_dir = tmp_path
    config.db_path = tmp_path / "test.db"
    return config

@pytest.fixture
def mock_orch_config() -> OrchestratorConfig:
    return OrchestratorConfig(max_generations=2, workers=1, dask_scheduler_address=None)

def test_workflow_grand_mock(mock_config: MagicMock, mock_orch_config: OrchestratorConfig) -> None:
    """
    Simulate a full run where each phase updates the state as expected.
    """
    with patch('mlip_autopipec.orchestration.manager.DatabaseManager') as MockDB, \
         patch('mlip_autopipec.orchestration.manager.TaskQueue') as MockTQ, \
         patch('mlip_autopipec.orchestration.manager.Dashboard') as MockDash:

        # Configure mock DB to return an integer for count()
        MockDB.return_value.count.return_value = 100

        manager = WorkflowManager(mock_config, mock_orch_config)

        # Mock Phase Methods to simulate work and state transitions

        def exploration_side_effect() -> None:
            # Simulate generating candidates
            manager.state.status = "dft"

        def dft_side_effect() -> None:
            # Simulate DFT calculations
            manager.state.status = "training"

        def training_side_effect() -> None:
            # Simulate Training
            manager.state.status = "inference"

        def inference_side_effect() -> None:
            # Simulate Inference and decision to continue
            manager.state.status = "idle"
            manager.state.current_generation += 1

        manager._run_exploration_phase = MagicMock(side_effect=exploration_side_effect) # type: ignore
        manager._run_dft_phase = MagicMock(side_effect=dft_side_effect) # type: ignore
        manager._run_training_phase = MagicMock(side_effect=training_side_effect) # type: ignore
        manager._run_inference_phase = MagicMock(side_effect=inference_side_effect) # type: ignore

        # Run
        manager.run()

        # Assertions
        assert manager.state.current_generation == 2
        assert manager._run_exploration_phase.call_count == 2
        assert manager._run_dft_phase.call_count == 2
        assert manager._run_training_phase.call_count == 2
        assert manager._run_inference_phase.call_count == 2

        # Verify dashboard was updated
        assert manager.dashboard.update.call_count >= 2  # type: ignore

        # Verify state file persisted
        state_file = mock_config.working_dir / "workflow_state.json"
        assert state_file.exists()
        final_state = json.loads(state_file.read_text())
        assert final_state['current_generation'] == 2
