from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.config.models import (
    MLIPConfig,
    TargetSystem,
    WorkflowConfig,
)
from mlip_autopipec.config.schemas.core import RuntimeConfig
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.domain_models.state import WorkflowPhase, WorkflowState
from mlip_autopipec.orchestration.workflow import WorkflowManager


@pytest.fixture
def mock_config(tmp_path):
    # Create fake UPF file
    (tmp_path / "Fe.UPF").touch()

    return MLIPConfig(
        target_system=TargetSystem(name="Test", elements=["Fe"], composition={"Fe": 1.0}),
        dft=DFTConfig(
            pseudopotential_dir=tmp_path, ecutwfc=30, kspacing=0.05, command="pw.x"
        ),
        workflow=WorkflowConfig(max_generations=2),
        runtime=RuntimeConfig(work_dir=tmp_path / "_work", database_path="mlip.db"),
    )


@patch("mlip_autopipec.orchestration.workflow.TaskQueue")
@patch("mlip_autopipec.orchestration.workflow.DatabaseManager")
def test_workflow_manager_initialization(mock_db, mock_tq, mock_config):
    manager = WorkflowManager(mock_config, work_dir=mock_config.runtime.work_dir)
    assert isinstance(manager.state, WorkflowState)
    assert manager.state.cycle_index == 0
    mock_tq.assert_called()


@patch("mlip_autopipec.orchestration.workflow.TrainingPhase")
@patch("mlip_autopipec.orchestration.workflow.DFTPhase")
@patch("mlip_autopipec.orchestration.workflow.SelectionPhase")
@patch("mlip_autopipec.orchestration.workflow.ExplorationPhase")
@patch("mlip_autopipec.orchestration.workflow.TaskQueue")
@patch("mlip_autopipec.orchestration.workflow.DatabaseManager")
def test_run_cycle_0(mock_db, mock_tq, mock_exp, mock_sel, mock_dft, mock_train, mock_config):
    """Test Cycle 0 execution (Cold Start) - Selection skipped."""
    manager = WorkflowManager(mock_config, work_dir=mock_config.runtime.work_dir)

    manager.run_cycle()

    # Verify phases executed
    mock_exp.assert_called_with(manager)
    mock_exp.return_value.execute.assert_called_once()

    # Selection skipped (cycle 0)
    mock_sel.assert_not_called()

    mock_dft.assert_called_with(manager)
    mock_dft.return_value.execute.assert_called_once()

    mock_train.assert_called_with(manager)
    mock_train.return_value.execute.assert_called_once()


@patch("mlip_autopipec.orchestration.workflow.TrainingPhase")
@patch("mlip_autopipec.orchestration.workflow.DFTPhase")
@patch("mlip_autopipec.orchestration.workflow.SelectionPhase")
@patch("mlip_autopipec.orchestration.workflow.ExplorationPhase")
@patch("mlip_autopipec.orchestration.workflow.TaskQueue")
@patch("mlip_autopipec.orchestration.workflow.DatabaseManager")
def test_run_cycle_1_with_potential(mock_db, mock_tq, mock_exp, mock_sel, mock_dft, mock_train, mock_config):
    """Test Cycle 1 execution (Active Learning) - Selection included."""
    manager = WorkflowManager(mock_config, work_dir=mock_config.runtime.work_dir)
    manager.state.cycle_index = 1
    manager.state.latest_potential_path = Path("fake.yace")

    manager.run_cycle()

    mock_exp.assert_called_with(manager)
    mock_exp.return_value.execute.assert_called_once()

    mock_sel.assert_called_with(manager)
    mock_sel.return_value.execute.assert_called_once()

    mock_dft.assert_called_with(manager)
    mock_dft.return_value.execute.assert_called_once()

    mock_train.assert_called_with(manager)
    mock_train.return_value.execute.assert_called_once()


@patch("mlip_autopipec.orchestration.workflow.TaskQueue")
@patch("mlip_autopipec.orchestration.workflow.DatabaseManager")
def test_run_loop(mock_db, mock_tq, mock_config):
    manager = WorkflowManager(mock_config, work_dir=mock_config.runtime.work_dir)

    # Mock run_cycle to avoid actual execution
    manager.run_cycle = MagicMock()

    manager.run()

    # Should run 2 cycles (0 and 1) because max_generations=2
    assert manager.run_cycle.call_count == 2
    assert manager.state.cycle_index == 2
    mock_tq.return_value.shutdown.assert_called_once()

@patch("mlip_autopipec.orchestration.workflow.TrainingPhase")
@patch("mlip_autopipec.orchestration.workflow.DFTPhase")
@patch("mlip_autopipec.orchestration.workflow.SelectionPhase")
@patch("mlip_autopipec.orchestration.workflow.ExplorationPhase")
@patch("mlip_autopipec.orchestration.workflow.TaskQueue")
@patch("mlip_autopipec.orchestration.workflow.DatabaseManager")
def test_resume_from_selection(mock_db, mock_tq, mock_exp, mock_sel, mock_dft, mock_train, mock_config):
    """Test resumption from Selection phase."""
    manager = WorkflowManager(mock_config, work_dir=mock_config.runtime.work_dir)
    manager.state.cycle_index = 1
    manager.state.latest_potential_path = Path("fake.yace")
    # Set state to SELECTION
    manager.state.current_phase = WorkflowPhase.SELECTION

    manager.run_cycle()

    # Exploration should be SKIPPED
    mock_exp.assert_not_called()

    # Selection should run
    mock_sel.assert_called_with(manager)

    # Rest should run
    mock_dft.assert_called()
    mock_train.assert_called()
