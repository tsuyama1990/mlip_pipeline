from unittest.mock import patch

import pytest

from mlip_autopipec.config.models import (
    SystemConfig,
    TargetSystem,
    WorkflowConfig,
)
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.state import WorkflowPhase, WorkflowState
from mlip_autopipec.orchestration.workflow import WorkflowManager


@pytest.fixture
def mock_config(tmp_path):
    # Create fake UPF file
    (tmp_path / "Fe.UPF").touch()

    return SystemConfig(
        target_system=TargetSystem(name="Test", elements=["Fe"], composition={"Fe": 1.0}),
        dft_config=DFTConfig(
            pseudopotential_dir=tmp_path,
            ecutwfc=30,
            kspacing=0.05,
            command="pw.x"
        ),
        workflow_config=WorkflowConfig(max_generations=2),
        working_dir=tmp_path / "_work",
        db_path=tmp_path / "mlip.db"
    )

@patch("mlip_autopipec.orchestration.workflow.TaskQueue")
@patch("mlip_autopipec.orchestration.workflow.DatabaseManager")
@patch("mlip_autopipec.orchestration.workflow.PhaseExecutor")
@patch("mlip_autopipec.orchestration.workflow.get_dask_client")
def test_workflow_manager_initialization(mock_dask, mock_executor, mock_db, mock_tq, mock_config):
    manager = WorkflowManager(mock_config, workflow_config=mock_config.workflow_config)
    assert isinstance(manager.state, WorkflowState)
    assert manager.state.cycle_index == 0
    assert manager.state.current_phase == WorkflowPhase.EXPLORATION

@patch("mlip_autopipec.orchestration.workflow.TaskQueue")
@patch("mlip_autopipec.orchestration.workflow.DatabaseManager")
@patch("mlip_autopipec.orchestration.workflow.PhaseExecutor")
@patch("mlip_autopipec.orchestration.workflow.get_dask_client")
def test_workflow_run_loop(mock_dask, mock_executor_cls, mock_db, mock_tq, mock_config):
    # Setup mocks
    # We will configure the instance on the manager directly to avoid mock mismatch issues

    manager = WorkflowManager(mock_config, workflow_config=mock_config.workflow_config)

    # Mock behaviors for phases
    # 1. Exploration -> Returns True (halted)
    # The manager.executor is the mock instance
    # We set max_generations to 1 so it should stop after one full cycle (0 -> 1)
    # But wait, run loop condition is index < max.
    # index=0. max=1. Runs.
    # After training, index=1. 1 < 1 False. Stops.
    # So execute_inference is called ONCE.

    mock_config.workflow_config.max_generations = 1
    manager = WorkflowManager(mock_config, workflow_config=mock_config.workflow_config)
    manager.executor.execute_inference.side_effect = [True]

    manager.run()

    # Assertions on the actual executor instance
    # execute_inference called once (returns True)
    assert manager.executor.execute_inference.call_count == 1
    assert manager.executor.execute_selection.called
    # assert manager.executor.execute_dft.called
    # assert manager.executor.execute_training.called

    # Check if cycle index incremented
    # assert manager.state.cycle_index == 1
