"""Tests for WorkflowManager and StateManager."""

from pathlib import Path

import pytest

from mlip_autopipec.domain_models.config import Config, PotentialConfig
from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState
from mlip_autopipec.orchestration.state_manager import StateManager
from mlip_autopipec.orchestration.workflow import WorkflowManager


@pytest.fixture
def mock_config(tmp_path: Path) -> Config:
    return Config(
        project_name="test",
        potential=PotentialConfig(elements=["Si"], cutoff=5.0),
        orchestrator={"state_file": tmp_path / "state.json"}  # type: ignore
    )


def test_state_manager_initialization(tmp_path: Path) -> None:
    """Test StateManager load/init."""
    path = tmp_path / "new_state.json"
    manager = StateManager(path)

    state = manager.load_or_initialize()
    assert state.cycle_index == 0
    assert state.current_phase == WorkflowPhase.INITIALIZATION


def test_state_manager_save_load(tmp_path: Path) -> None:
    """Test StateManager persistence."""
    path = tmp_path / "saved_state.json"
    manager = StateManager(path)

    state = WorkflowState(cycle_index=2, current_phase=WorkflowPhase.TRAINING)
    manager.save(state)

    loaded = manager.load_or_initialize()
    assert loaded.cycle_index == 2
    assert loaded.current_phase == WorkflowPhase.TRAINING


def test_state_manager_load_corruption(tmp_path: Path) -> None:
    """Test corrupted state file."""
    path = tmp_path / "corrupt.json"
    path.write_text("{bad_json")

    manager = StateManager(path)
    state = manager.load_or_initialize()
    assert state.cycle_index == 0  # Should reset

    # Check if backup was created
    backups = list(tmp_path.glob("corrupt.bak.*.json"))
    assert len(backups) == 1


def test_workflow_run(mock_config: Config, tmp_path: Path) -> None:
    """Test WorkflowManager run."""
    manager = WorkflowManager(mock_config)
    manager.run()

    assert manager.state.current_phase == WorkflowPhase.EXPLORATION
    # Verify save happened via StateManager
    saved_state = manager.state_manager.load_or_initialize()
    assert saved_state.current_phase == WorkflowPhase.EXPLORATION
