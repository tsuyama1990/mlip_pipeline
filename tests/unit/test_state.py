from pathlib import Path

from mlip_autopipec.core.state import StateManager


def test_state_manager_initialization(tmp_path: Path) -> None:
    state_path = tmp_path / "workflow_state.json"
    manager = StateManager(state_path)

    # File is not created until save is called
    assert not state_path.exists()
    assert manager.state.current_cycle == 0
    assert manager.state.status == "IDLE"


def test_state_save(tmp_path: Path) -> None:
    state_path = tmp_path / "workflow_state.json"
    manager = StateManager(state_path)
    manager.save()
    assert state_path.exists()


def test_state_updates(tmp_path: Path) -> None:
    state_path = tmp_path / "workflow_state.json"
    manager = StateManager(state_path)

    manager.update_cycle(2)
    assert manager.state.current_cycle == 2

    manager.update_status("RUNNING")
    assert manager.state.status == "RUNNING"

    # Reload from file
    manager2 = StateManager(state_path)
    assert manager2.state.current_cycle == 2
    assert manager2.state.status == "RUNNING"
