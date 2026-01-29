
from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState


def test_workflow_state_defaults() -> None:
    """Test default workflow state."""
    state = WorkflowState()
    assert state.cycle_index == 0
    assert state.current_phase == WorkflowPhase.EXPLORATION
    assert state.dataset_path is None
    assert not state.is_halted


def test_workflow_state_transition() -> None:
    """Test manual state transition."""
    state = WorkflowState()
    state.current_phase = WorkflowPhase.SELECTION
    state.cycle_index = 1

    assert state.current_phase == WorkflowPhase.SELECTION
    assert state.cycle_index == 1


def test_workflow_state_meta() -> None:
    """Test metadata storage."""
    state = WorkflowState(meta={"last_rmse": 0.001})
    assert state.meta["last_rmse"] == 0.001
