from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from mlip_autopipec.orchestration.state import StateManager
from mlip_autopipec.domain_models.workflow import WorkflowState, WorkflowPhase

@pytest.fixture
def mock_state():
    return WorkflowState(
        project_name="Test",
        dataset_path=Path("data.pckl"),
        current_phase=WorkflowPhase.EXPLORATION
    )

def test_state_atomic_save(tmp_path, mock_state):
    manager = StateManager(tmp_path)
    manager.save(mock_state)

    assert (tmp_path / "workflow_state.json").exists()
    assert not (tmp_path / "workflow_state.json.tmp").exists()

def test_state_save_failure_cleanup(tmp_path, mock_state):
    manager = StateManager(tmp_path)

    # Mock rename to fail
    with patch("pathlib.Path.rename", side_effect=OSError("Rename failed")):
        with pytest.raises(OSError):
            manager.save(mock_state)

    # Tmp file should be cleaned up
    assert not (tmp_path / "workflow_state.json.tmp").exists()
    # Original file (if any) remains?
    # Here we started with nothing, so nothing.

def test_state_load(tmp_path, mock_state):
    manager = StateManager(tmp_path)
    manager.save(mock_state)

    loaded = manager.load()
    assert loaded is not None
    assert loaded.project_name == mock_state.project_name
