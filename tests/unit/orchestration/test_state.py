from mlip_autopipec.domain_models.state import WorkflowPhase, WorkflowState


def test_workflow_state_initialization():
    state = WorkflowState()
    assert state.cycle_index == 0
    assert state.current_phase == WorkflowPhase.EXPLORATION
    assert state.active_tasks == []
    assert state.latest_potential_path is None


def test_workflow_state_transitions():
    state = WorkflowState()
    state.current_phase = WorkflowPhase.SELECTION
    assert state.current_phase == "Selection"

    state.cycle_index += 1
    assert state.cycle_index == 1


def test_workflow_state_serialization():
    state = WorkflowState(
        cycle_index=2, current_phase=WorkflowPhase.TRAINING, active_tasks=["task1"]
    )
    json_str = state.model_dump_json()

    loaded_state = WorkflowState.model_validate_json(json_str)
    assert loaded_state.cycle_index == 2
    assert loaded_state.current_phase == WorkflowPhase.TRAINING
    assert loaded_state.active_tasks == ["task1"]
