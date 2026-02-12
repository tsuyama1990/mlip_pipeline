from pathlib import Path

from mlip_autopipec.core.state_manager import StateManager
from mlip_autopipec.domain_models.datastructures import WorkflowState
from mlip_autopipec.domain_models.enums import TaskStatus


def test_load_state_missing(tmp_path: Path) -> None:
    sm = StateManager(tmp_path)
    assert sm.load_state() is None


def test_save_load_cycle(tmp_path: Path) -> None:
    sm = StateManager(tmp_path)
    state = WorkflowState(iteration=5, status=TaskStatus.RUNNING)
    sm.save_state(state)

    assert (tmp_path / "workflow_state.json").exists()

    loaded = sm.load_state()
    assert loaded is not None
    assert loaded.iteration == 5
    assert loaded.status == TaskStatus.RUNNING


def test_cleanup(tmp_path: Path) -> None:
    sm = StateManager(tmp_path)
    (tmp_path / "foo.tmp").touch()
    (tmp_path / "bar.tmp").touch()
    (tmp_path / "other.txt").touch()

    sm.cleanup()

    assert not (tmp_path / "foo.tmp").exists()
    assert not (tmp_path / "bar.tmp").exists()
    assert (tmp_path / "other.txt").exists()
