import json
from pathlib import Path

from mlip_autopipec.orchestration.state import StateManager

from mlip_autopipec.domain_models.workflow import WorkflowState


def test_state_manager_save_load(temp_dir: Path) -> None:
    state_file = temp_dir / "state.json"
    manager = StateManager(state_file)

    # Initial state
    initial_state = WorkflowState(iteration=1)
    manager.save(initial_state)

    assert state_file.exists()

    # Load state
    loaded_state = manager.load()
    assert loaded_state is not None
    assert loaded_state.iteration == 1


def test_state_manager_atomic_write(temp_dir: Path) -> None:
    state_file = temp_dir / "state.json"
    manager = StateManager(state_file)
    state = WorkflowState(iteration=2)

    manager.save(state)
    assert state_file.exists()

    # Check content
    with state_file.open() as f:
        data = json.load(f)
    assert data["iteration"] == 2


def test_state_manager_load_missing(temp_dir: Path) -> None:
    state_file = temp_dir / "missing.json"
    manager = StateManager(state_file)

    assert manager.load() is None
