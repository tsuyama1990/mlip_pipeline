import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState, CandidateStructure
from mlip_autopipec.orchestration.manager import WorkflowManager

@pytest.fixture
def mock_config():
    config = MagicMock(spec=Config)
    config.orchestrator = MagicMock()
    config.orchestrator.max_iterations = 2
    config.orchestrator.validation_frequency = 1
    config.logging = MagicMock()
    config.logging.file_path = Path("test.log")
    config.training = MagicMock()
    config.training.initial_potential = None
    config.validation = MagicMock()
    config.validation.report_path = Path("validation_report.html")
    # Fix: Ensure all fields accessed by Manager are mocked
    config.structure_gen = MagicMock()
    config.potential = MagicMock()
    config.lammps = MagicMock()
    config.dft = MagicMock()
    config.md = MagicMock()

    return config

@pytest.fixture
def mock_state():
    return WorkflowState(
        project_name="Test",
        dataset_path=Path("data.pckl"),
        current_phase=WorkflowPhase.EXPLORATION,
        generation=0
    )

def test_manager_initialization(mock_config, tmp_path):
    with patch("mlip_autopipec.orchestration.manager.StateManager"):
        manager = WorkflowManager(mock_config, work_dir=tmp_path)
        assert manager.work_dir == tmp_path

def test_manager_step_exploration(mock_config, mock_state, tmp_path):
    with patch("mlip_autopipec.orchestration.manager.StateManager") as MockStateMgr:
        MockStateMgr.return_value.load.return_value = mock_state

        manager = WorkflowManager(mock_config, work_dir=tmp_path)
        manager.state = mock_state

        # Mock explore method
        manager.explore = MagicMock(return_value=True) # Returns True if halt detected

        manager.step()

        assert manager.explore.called
        assert manager.state.current_phase == WorkflowPhase.SELECTION
        assert MockStateMgr.return_value.save.called

def test_manager_step_selection(mock_config, mock_state, tmp_path):
    mock_state.current_phase = WorkflowPhase.SELECTION

    with patch("mlip_autopipec.orchestration.manager.StateManager") as MockStateMgr:
        MockStateMgr.return_value.load.return_value = mock_state

        manager = WorkflowManager(mock_config, work_dir=tmp_path)
        manager.state = mock_state

        manager.select = MagicMock(return_value=[MagicMock(spec=CandidateStructure)])

        manager.step()

        assert manager.select.called
        assert manager.state.current_phase == WorkflowPhase.CALCULATION

def test_manager_completion(mock_config, mock_state, tmp_path):
    mock_state.generation = 2
    mock_config.orchestrator.max_iterations = 2

    with patch("mlip_autopipec.orchestration.manager.StateManager"):
        manager = WorkflowManager(mock_config, work_dir=tmp_path)
        manager.state = mock_state

        should_continue = manager.step()
        assert not should_continue

def test_manager_validation_report_check(mock_config, mock_state, tmp_path):
    # Ensure validation report path is checked or used
    # In validate method, we log report path.
    # Here we can just check if report generation is attempted.

    mock_state.current_phase = WorkflowPhase.VALIDATION
    mock_state.latest_potential_path = Path("pot.yace")

    with patch("mlip_autopipec.orchestration.manager.StateManager") as MockStateMgr, \
         patch("mlip_autopipec.orchestration.manager.ValidationRunner") as MockValRunner, \
         patch("mlip_autopipec.orchestration.manager.StructureGenFactory"):

        MockStateMgr.return_value.load.return_value = mock_state
        manager = WorkflowManager(mock_config, work_dir=tmp_path)
        manager.state = mock_state

        # Mock validation result
        MockValRunner.return_value.validate.return_value = MagicMock(overall_status="PASS")

        manager.step()

        # Assert ValidationRunner called with config
        MockValRunner.assert_called_with(
            val_config=mock_config.validation,
            pot_config=mock_config.potential,
            potential_path=mock_state.latest_potential_path
        )

        # Report verification is implicitly done by ValidationRunner (which we mocked),
        # but the test ensures we pass the config correctly.
