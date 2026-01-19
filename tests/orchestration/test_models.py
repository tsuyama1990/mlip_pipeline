import pytest
from pydantic import ValidationError

from mlip_autopipec.orchestration.models import DashboardData, OrchestratorConfig, WorkflowState


def test_workflow_state_valid():
    state = WorkflowState(current_generation=0, status="idle")
    assert state.current_generation == 0
    assert state.status == "idle"
    assert state.pending_tasks == []


def test_workflow_state_valid_statuses():
    for status in ["idle", "dft", "training", "inference", "extraction"]:
        state = WorkflowState(status=status)
        assert state.status == status


def test_workflow_state_invalid_status():
    with pytest.raises(ValidationError):
        WorkflowState(status="invalid_status")


def test_workflow_state_default_pending():
    state = WorkflowState()
    assert state.pending_tasks == []


def test_orchestrator_config_defaults():
    config = OrchestratorConfig()
    assert config.max_generations == 5
    assert config.dask_scheduler_address is None
    assert config.workers == 4


def test_orchestrator_config_custom():
    config = OrchestratorConfig(
        max_generations=10, dask_scheduler_address="tcp://localhost:8786", workers=8
    )
    assert config.max_generations == 10
    assert config.dask_scheduler_address == "tcp://localhost:8786"
    assert config.workers == 8


def test_orchestrator_config_validation():
    with pytest.raises(ValidationError):
        OrchestratorConfig(max_generations=0)  # Must be >= 1

    with pytest.raises(ValidationError):
        OrchestratorConfig(workers=0)  # Must be >= 1


def test_dashboard_data_defaults():
    data = DashboardData()
    assert data.generations == []
    assert data.rmse_values == []
    assert data.structure_counts == []
    assert data.status == "Unknown"


def test_dashboard_data_valid():
    data = DashboardData(
        generations=[0, 1], rmse_values=[0.5, 0.4], structure_counts=[100, 200], status="training"
    )
    assert len(data.generations) == 2
    assert data.status == "training"
