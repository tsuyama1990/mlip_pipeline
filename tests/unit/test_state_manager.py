import json
from pathlib import Path

import pytest

from mlip_autopipec.core.state_manager import StateManager
from mlip_autopipec.domain_models import WorkflowState, WorkflowStatus


def test_state_manager_save_load(tmp_path: Path) -> None:
    state_file = tmp_path / "workflow_state.json"
    manager = StateManager(state_file)

    initial_state = WorkflowState(iteration=1, status=WorkflowStatus.EXPLORATION)
    manager.save(initial_state)

    assert state_file.exists()

    loaded_state = manager.load()
    assert loaded_state.iteration == 1
    assert loaded_state.status == WorkflowStatus.EXPLORATION


def test_state_manager_atomic_write(tmp_path: Path) -> None:
    state_file = tmp_path / "workflow_state.json"
    manager = StateManager(state_file)

    initial_state = WorkflowState(iteration=2, status=WorkflowStatus.TRAINING)
    manager.save(initial_state)

    with state_file.open("r") as f:
        data = json.load(f)
    assert data["iteration"] == 2
    assert data["status"] == "TRAINING"


def test_state_manager_load_non_existent(tmp_path: Path) -> None:
    state_file = tmp_path / "non_existent.json"
    manager = StateManager(state_file)
    state = manager.load()

    assert state.iteration == 0
    assert state.status == WorkflowStatus.IDLE


def test_state_manager_load_corrupted(tmp_path: Path) -> None:
    state_file = tmp_path / "corrupted.json"
    state_file.write_text("{invalid json")
    manager = StateManager(state_file)
    with pytest.raises(RuntimeError):
        manager.load()


def test_state_manager_cleanup(tmp_path: Path) -> None:
    state_file = tmp_path / "test.json"
    temp_file = state_file.with_suffix(".tmp")
    temp_file.touch()
    manager = StateManager(state_file)
    manager.cleanup()
    assert not temp_file.exists()
