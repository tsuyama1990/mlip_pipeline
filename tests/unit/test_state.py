import json
from pathlib import Path

from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.orchestration.state import StateManager


def test_workflow_state_initialization() -> None:
    # iteration=0 and current_potential_path=None are defaults in WorkflowState
    state = WorkflowState(iteration=0, current_potential_path=None)
    assert state.iteration == 0
    assert state.current_potential_path is None
    assert state.history == []


def test_workflow_state_update() -> None:
    state = WorkflowState(iteration=0, current_potential_path=None)
    state.increment_iteration()
    assert state.iteration == 1

    p = Path("pot.yace")
    state.update_potential(p)
    assert state.current_potential_path == p


def test_state_persistence(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    manager = StateManager(state_file)

    # Save initial state
    state = WorkflowState(iteration=5, current_potential_path=None)
    manager.save(state)

    assert state_file.exists()

    # Load back
    loaded_state = manager.load()
    assert loaded_state.iteration == 5

    # Atomic write verification (rudimentary)
    # Ideally we'd test that .tmp exists during write, but that's hard to catch.
    # We can check that the file contains valid JSON.
    with state_file.open("r") as f:
        data = json.load(f)
    assert data["iteration"] == 5
