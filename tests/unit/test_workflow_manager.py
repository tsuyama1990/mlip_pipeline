from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.orchestration.workflow import WorkflowManager


@pytest.fixture
def mock_config():
    conf = MagicMock()
    conf.working_dir = Path("/tmp")
    conf.db_path = Path("/tmp/db.sqlite")
    conf.training_config = MagicMock()
    return conf

@pytest.fixture
def mock_orch_config():
    conf = MagicMock()
    conf.max_generations = 2
    conf.workers = 1
    conf.dask_scheduler_address = None
    return conf

from pathlib import Path

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.orchestration.models import OrchestratorConfig


def test_workflow_manager_init(tmp_path):
    """Test initialization of WorkflowManager."""
    config = MagicMock(spec=SystemConfig)
    config.working_dir = tmp_path
    config.db_path = tmp_path / "test.db"

    # Mocking components to avoid real I/O
    with patch("mlip_autopipec.orchestration.workflow.DatabaseManager"), \
         patch("mlip_autopipec.orchestration.workflow.TaskQueue"), \
         patch("mlip_autopipec.orchestration.workflow.Dashboard"):

        orch_config = OrchestratorConfig(max_generations=1)
        manager = WorkflowManager(config, orch_config)

        assert manager.state.current_generation == 0
        assert manager.state.status == "idle"

def test_workflow_dispatch_logic():
    """Test state transition logic dispatching."""
    with patch("mlip_autopipec.orchestration.workflow.DatabaseManager"), \
         patch("mlip_autopipec.orchestration.workflow.TaskQueue"), \
         patch("mlip_autopipec.orchestration.workflow.Dashboard"), \
         patch("mlip_autopipec.orchestration.workflow.PhaseExecutor") as MockExecutor:

        # Setup
        config = MagicMock(spec=SystemConfig)
        config.working_dir = Path("/tmp")
        orch_config = OrchestratorConfig(max_generations=1)

        manager = WorkflowManager(config, orch_config)
        executor_instance = MockExecutor.return_value

        # 1. Idle -> DFT
        manager.state.status = "idle"
        manager._dispatch_phase()
        executor_instance.execute_exploration.assert_called_once()
        assert manager.state.status == "dft"

        # 2. DFT -> Training
        manager.state.status = "dft"
        manager._dispatch_phase()
        executor_instance.execute_dft.assert_called_once()
        assert manager.state.status == "training"

        # 3. Training -> Inference
        manager.state.status = "training"
        manager._dispatch_phase()
        executor_instance.execute_training.assert_called_once()
        assert manager.state.status == "inference"

        # 4. Inference -> Idle (Converged)
        manager.state.status = "inference"
        executor_instance.execute_inference.return_value = False # Not active learning
        manager._dispatch_phase()
        executor_instance.execute_inference.assert_called()
        assert manager.state.status == "idle"
        assert manager.state.current_generation == 1

        # 5. Inference -> DFT (Active Learning)
        manager.state.status = "inference"
        executor_instance.execute_inference.return_value = True # Active learning triggered
        manager._dispatch_phase()
        assert manager.state.status == "dft"
        # Generation should NOT increment
        assert manager.state.current_generation == 1
