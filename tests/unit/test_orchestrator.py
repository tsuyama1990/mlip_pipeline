import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from mlip_autopipec.domain_models.config import Config, ACEConfig, PotentialConfig
from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.domain_models.dynamics import LammpsResult
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.structure import Structure
import numpy as np

@pytest.fixture
def mock_config():
    config = MagicMock(spec=Config)
    config.project_name = "Test"
    config.orchestrator = MagicMock()
    config.orchestrator.max_iterations = 2
    config.orchestrator.uncertainty_threshold = 0.5
    config.orchestrator.validation_frequency = 1
    config.training = MagicMock()
    config.training.initial_potential = None
    config.potential = MagicMock(spec=PotentialConfig)
    config.potential.ace_params = MagicMock(spec=ACEConfig)
    return config

@pytest.fixture
def mock_state():
    return WorkflowState(
        project_name="Test",
        dataset_path=Path("data/data.pckl"),
        current_phase=WorkflowPhase.EXPLORATION,
        generation=0
    )

@pytest.fixture
def dummy_structure():
    return Structure(
        symbols=["Si"], positions=np.array([[0,0,0]]), cell=np.eye(3), pbc=(True,True,True)
    )

def test_orchestrator_initialization(mock_config, tmp_path):
    with patch("mlip_autopipec.orchestration.orchestrator.StateManager") as MockStateMgr, \
         patch("mlip_autopipec.orchestration.orchestrator.ExplorationPhase"), \
         patch("mlip_autopipec.orchestration.orchestrator.SelectionPhase"), \
         patch("mlip_autopipec.orchestration.orchestrator.CalculationPhase"), \
         patch("mlip_autopipec.orchestration.orchestrator.TrainingPhase"), \
         patch("mlip_autopipec.orchestration.orchestrator.ValidationPhase"):

        # Simulate first run: load returns None
        MockStateMgr.return_value.load.return_value = None

        orchestrator = Orchestrator(mock_config, work_dir=tmp_path)
        assert orchestrator.work_dir == tmp_path
        assert orchestrator.state.current_phase == WorkflowPhase.EXPLORATION
        assert orchestrator.state.project_name == "Test"

def test_orchestrator_step_exploration_halt(mock_config, mock_state, tmp_path, dummy_structure):
    with patch("mlip_autopipec.orchestration.orchestrator.StateManager") as MockStateMgr, \
         patch("mlip_autopipec.orchestration.orchestrator.ExplorationPhase") as MockExploration, \
         patch("mlip_autopipec.orchestration.orchestrator.SelectionPhase"), \
         patch("mlip_autopipec.orchestration.orchestrator.CalculationPhase"), \
         patch("mlip_autopipec.orchestration.orchestrator.TrainingPhase"), \
         patch("mlip_autopipec.orchestration.orchestrator.ValidationPhase"):

        MockStateMgr.return_value.load.return_value = mock_state

        # Mock Exploration Result (Halt)
        MockExploration.return_value.execute.return_value = LammpsResult(
            job_id="test", status=JobStatus.COMPLETED, work_dir=tmp_path, duration_seconds=1,
            log_content="", max_gamma=1.0, # > 0.5 threshold
            final_structure=dummy_structure,
            trajectory_path=tmp_path / "traj.lammpstrj"
        )

        orchestrator = Orchestrator(mock_config, work_dir=tmp_path)
        orchestrator.state = mock_state

        orchestrator.step()

        assert MockExploration.return_value.execute.called
        assert orchestrator.state.current_phase == WorkflowPhase.SELECTION
        assert MockStateMgr.return_value.save.called

def test_orchestrator_step_exploration_converged(mock_config, mock_state, tmp_path, dummy_structure):
    with patch("mlip_autopipec.orchestration.orchestrator.StateManager") as MockStateMgr, \
         patch("mlip_autopipec.orchestration.orchestrator.ExplorationPhase") as MockExploration, \
         patch("mlip_autopipec.orchestration.orchestrator.SelectionPhase"), \
         patch("mlip_autopipec.orchestration.orchestrator.CalculationPhase"), \
         patch("mlip_autopipec.orchestration.orchestrator.TrainingPhase"), \
         patch("mlip_autopipec.orchestration.orchestrator.ValidationPhase"):

        MockStateMgr.return_value.load.return_value = mock_state

        # Mock Exploration Result (Converged)
        MockExploration.return_value.execute.return_value = LammpsResult(
            job_id="test", status=JobStatus.COMPLETED, work_dir=tmp_path, duration_seconds=1,
            log_content="", max_gamma=0.1, # < 0.5 threshold
            final_structure=dummy_structure,
            trajectory_path=tmp_path / "traj.lammpstrj"
        )

        orchestrator = Orchestrator(mock_config, work_dir=tmp_path)

        should_continue = orchestrator.step()

        assert not should_continue
        assert orchestrator.state.current_phase == WorkflowPhase.EXPLORATION # Didn't change
