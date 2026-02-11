from pathlib import Path

from mlip_autopipec.core.state_manager import StateManager


def test_state_manager_load_missing(tmp_path: Path) -> None:
    sm = StateManager(tmp_path / "missing.json")
    state = sm.load()
    assert state.current_iteration == 0
    assert not state.completed_tasks
