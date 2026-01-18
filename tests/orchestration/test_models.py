import pytest
from pydantic import ValidationError

from mlip_autopipec.orchestration.models import DashboardData, OrchestratorConfig, WorkflowState


def test_workflow_state_defaults() -> None:
    state = WorkflowState()
    assert state.current_generation == 0
    assert state.status == "idle"
    assert state.pending_tasks == []

def test_workflow_state_valid() -> None:
    state = WorkflowState(
        current_generation=2,
        status="training",
        pending_tasks=["task1", "task2"]
    )
    assert state.current_generation == 2
    assert state.status == "training"
    assert len(state.pending_tasks) == 2

def test_workflow_state_invalid_status() -> None:
    with pytest.raises(ValidationError):
        WorkflowState(status="invalid_status")

def test_orchestrator_config_defaults() -> None:
    config = OrchestratorConfig()
    assert config.max_generations == 5
    assert config.workers == 4
    assert config.dask_scheduler_address is None

def test_orchestrator_config_validation() -> None:
    with pytest.raises(ValidationError):
        OrchestratorConfig(workers=0)

    with pytest.raises(ValidationError):
        OrchestratorConfig(max_generations=0)

def test_dashboard_data() -> None:
    data = DashboardData(
        generations=[0, 1],
        rmse_values=[0.5, 0.4],
        structure_counts=[100, 200],
        status="training"
    )
    assert len(data.generations) == 2
    assert data.status == "training"
