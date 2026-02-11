from pathlib import Path

from mlip_autopipec.core.state_manager import StateManager


def test_state_manager_cleanup(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    sm = StateManager(state_file)

    # Create a dummy .tmp file
    temp_file = state_file.with_suffix('.tmp')
    temp_file.touch()
    assert temp_file.exists()

    sm.cleanup()
    assert not temp_file.exists()

def test_state_manager_cleanup_no_file(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    sm = StateManager(state_file)

    # cleanup should not raise error if file doesn't exist
    sm.cleanup()
