from pathlib import Path

from mlip_autopipec.core.state_manager import StateManager
from mlip_autopipec.domain_models.datastructures import WorkflowState


def test_state_manager_save_load(tmp_path: Path) -> None:
    sm = StateManager(work_dir=tmp_path)
    state = WorkflowState(iteration=1)
    sm.save_state(state)

    assert (tmp_path / "workflow_state.json").exists()

    loaded_state = sm.load_state()
    assert loaded_state is not None
    assert loaded_state.iteration == 1


def test_state_manager_load_empty(tmp_path: Path) -> None:
    sm = StateManager(work_dir=tmp_path)
    loaded_state = sm.load_state()
    assert loaded_state is None


def test_state_manager_atomic_cleanup(tmp_path: Path) -> None:
    sm = StateManager(work_dir=tmp_path)
    state = WorkflowState(iteration=2)
    sm.save_state(state)

    # Check no tmp files remain
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0
