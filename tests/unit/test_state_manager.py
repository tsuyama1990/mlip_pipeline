import pytest
from pathlib import Path
from mlip_autopipec.domain_models import WorkflowState, WorkflowStage
from mlip_autopipec.core.state_manager import StateManager

def test_workflow_state_defaults() -> None:
    state = WorkflowState()
    assert state.iteration == 0
    assert state.current_stage == WorkflowStage.EXPLORE
    assert state.latest_potential_path is None

def test_workflow_state_serialization(tmp_path: Path) -> None:
    dummy_path = tmp_path / "pot.yace"
    state = WorkflowState(iteration=1, current_stage=WorkflowStage.LABEL, latest_potential_path=dummy_path)
    json_str = state.model_dump_json()
    # Pydantic V2 model_dump_json produces compact JSON by default
    assert '"iteration":1' in json_str.replace(" ", "")
    assert '"current_stage":"label"' in json_str

    loaded = WorkflowState.model_validate_json(json_str)
    assert loaded.iteration == 1
    assert loaded.current_stage == WorkflowStage.LABEL
    assert loaded.latest_potential_path == dummy_path

def test_state_manager_persistence(tmp_path: Path) -> None:
    """
    Test StateManager save and load.
    """
    manager = StateManager(work_dir=tmp_path)
    state = WorkflowState(iteration=5)
    manager.save(state)

    assert (tmp_path / "workflow_state.json").exists()

    loaded_state = manager.load()
    assert loaded_state.iteration == 5
