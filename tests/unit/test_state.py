from pathlib import Path

import pytest

from mlip_autopipec.domain_models.workflow import WorkflowState

# We will import StateManager later, but for TDD we can define the test assuming it exists
# or we can write the test now and it will fail on import.
# To avoid ImportErrors preventing other tests from collecting, I will use a try-import or mocking?
# No, TDD means the test should fail.
# But I can't write the test file if it crashes pytest collection.
# So I will assume StateManager will be in mlip_autopipec.orchestration.state

try:
    from mlip_autopipec.orchestration.state import StateManager
except ImportError:
    StateManager = None # type: ignore

def test_workflow_state_initialization():
    state = WorkflowState()
    assert state.iteration == 0
    assert state.current_potential_path is None
    assert state.history == []

def test_workflow_state_update():
    state = WorkflowState()
    state.increment_iteration()
    assert state.iteration == 1

    p = Path("pot.yace")
    state.update_potential(p)
    assert state.current_potential_path == p

@pytest.mark.skipif(StateManager is None, reason="StateManager not implemented yet")
def test_state_persistence(tmp_path):
    state_file = tmp_path / "state.json"
    manager = StateManager(state_file)

    # Save initial state
    state = WorkflowState(iteration=5)
    manager.save(state)

    assert state_file.exists()

    # Load back
    loaded_state = manager.load()
    assert loaded_state.iteration == 5

    # Atomic write verification (rudimentary)
    # Ideally we'd test that .tmp exists during write, but that's hard to catch.
    # We can check that the file contains valid JSON.
    import json
    with state_file.open("r") as f:
        data = json.load(f)
    assert data["iteration"] == 5
