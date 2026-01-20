import pytest
from pydantic import ValidationError

from mlip_autopipec.orchestration.models import OrchestratorConfig, WorkflowState


def test_workflow_state_defaults() -> None:
    state = WorkflowState()
    assert state.current_generation == 0
    assert state.status == "idle"
    assert state.pending_tasks == []


def test_workflow_state_validation() -> None:
    # Test valid status
    state = WorkflowState(status="dft")
    assert state.status == "dft"

    # Test invalid status
    with pytest.raises(ValidationError):
        WorkflowState(status="invalid_status")  # type: ignore

    # Test extra fields forbidden
    with pytest.raises(ValidationError):
        WorkflowState(extra_field="fail")  # type: ignore


def test_orchestrator_config_defaults() -> None:
    config = OrchestratorConfig()
    assert config.max_generations == 5
    assert config.dask_scheduler_address is None
    assert config.workers == 4


def test_orchestrator_config_validation() -> None:
    # Test valid values
    config = OrchestratorConfig(max_generations=10, workers=8)
    assert config.max_generations == 10
    assert config.workers == 8

    # Test extra fields forbidden
    with pytest.raises(ValidationError):
        OrchestratorConfig(extra_field="fail")  # type: ignore

    # Test type validation
    with pytest.raises(ValidationError):
        OrchestratorConfig(workers="four")  # type: ignore
