from pathlib import Path

import pytest

from mlip_autopipec.core.state_manager import StateManager
from mlip_autopipec.core.exceptions import StateError
from mlip_autopipec.domain_models.inputs import ProjectState
from mlip_autopipec.domain_models.enums import TaskStatus


def test_save_and_load(tmp_path: Path) -> None:
    """Test saving and loading the state."""
    manager = StateManager(tmp_path)

    # Create initial state
    state = ProjectState(
        current_iteration=5,
        status=TaskStatus.RUNNING
    )

    # Save
    manager.save(state)
    assert (tmp_path / "workflow_state.json").exists()

    # Load
    loaded_state = manager.load()
    assert loaded_state.current_iteration == 5
    assert loaded_state.status == TaskStatus.RUNNING


def test_load_non_existent(tmp_path: Path) -> None:
    """Test loading when no state file exists (should return default)."""
    manager = StateManager(tmp_path)
    state = manager.load()
    assert state.current_iteration == 0
    assert state.status == TaskStatus.PENDING


def test_atomic_write(tmp_path: Path) -> None:
    """Test atomic write by checking for temporary file behavior (indirectly)."""
    manager = StateManager(tmp_path)
    state = ProjectState(current_iteration=1)

    # Save creates the file
    manager.save(state)
    assert (tmp_path / "workflow_state.json").exists()
    assert not (tmp_path / "workflow_state.tmp").exists() # Should be cleaned up


def test_corrupted_state_file(tmp_path: Path) -> None:
    """Test handling of corrupted JSON file."""
    manager = StateManager(tmp_path)
    state_file = tmp_path / "workflow_state.json"
    state_file.write_text("{invalid_json")

    with pytest.raises(StateError):
        manager.load()


def test_save_permission_error(tmp_path: Path) -> None:
    """Test handling of permission error during save."""
    # Create a read-only directory
    ro_dir = tmp_path / "read_only"
    ro_dir.mkdir()
    try:
        # Remove write permissions
        ro_dir.chmod(0o500)

        manager = StateManager(ro_dir)
        state = ProjectState()

        with pytest.raises(StateError):
            manager.save(state)
    finally:
        # Restore permissions to allow cleanup
        ro_dir.chmod(0o700)
