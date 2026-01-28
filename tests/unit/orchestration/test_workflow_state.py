from pathlib import Path

from mlip_autopipec.domain_models.state import WorkflowPhase, WorkflowState


def test_workflow_state_defaults():
    state = WorkflowState()
    assert state.cycle_index == 0
    assert state.current_phase == WorkflowPhase.EXPLORATION
    assert state.dataset_path is None
    assert state.halted_structures == []

def test_workflow_state_serialization():
    state = WorkflowState(
        cycle_index=1,
        current_phase=WorkflowPhase.SELECTION,
        dataset_path=Path("data.pckl"),
        halted_structures=[Path("run1.dump"), Path("run2.dump")]
    )

    json_str = state.model_dump_json()
    loaded = WorkflowState.model_validate_json(json_str)

    assert loaded.cycle_index == 1
    assert loaded.current_phase == WorkflowPhase.SELECTION
    assert loaded.dataset_path == Path("data.pckl")
    assert loaded.halted_structures == [Path("run1.dump"), Path("run2.dump")]
