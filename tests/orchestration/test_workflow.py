"""Tests for WorkflowManager."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState
from mlip_autopipec.orchestration.workflow import WorkflowManager


@pytest.fixture
def mock_config() -> Config:
    return Config(
        project_name="test",
        potential={"elements": ["Si"], "cutoff": 5.0}
    )


def test_workflow_initialization(tmp_path: Path, mock_config: Config) -> None:
    """Test initializing a new workflow."""
    manager = WorkflowManager(mock_config, state_path=tmp_path / "state.json")
    assert manager.state.cycle_index == 0
    assert manager.state.current_phase == WorkflowPhase.INITIALIZATION

    # Verify save works
    manager.save_state()
    assert (tmp_path / "state.json").exists()


def test_workflow_load_existing(tmp_path: Path, mock_config: Config) -> None:
    """Test loading existing state."""
    state_path = tmp_path / "state.json"
    initial_state = WorkflowState(cycle_index=5, current_phase=WorkflowPhase.TRAINING)

    # Save manually
    from mlip_autopipec.infrastructure import io
    io.save_json(initial_state.model_dump(mode="json"), state_path)

    manager = WorkflowManager(mock_config, state_path=state_path)
    assert manager.state.cycle_index == 5
    assert manager.state.current_phase == WorkflowPhase.TRAINING


def test_workflow_load_failure(tmp_path: Path, mock_config: Config) -> None:
    """Test handling of corrupted state file."""
    state_path = tmp_path / "state.json"
    state_path.write_text("{corrupted json")

    # Should log error and start fresh
    manager = WorkflowManager(mock_config, state_path=state_path)
    assert manager.state.cycle_index == 0


def test_workflow_save_failure(tmp_path: Path, mock_config: Config) -> None:
    """Test handling of save failure."""
    # Use a directory as file path to trigger error
    state_path = tmp_path / "directory"
    state_path.mkdir()

    manager = WorkflowManager(mock_config, state_path=state_path)
    # Should log error but not crash
    manager.save_state()

def test_workflow_run(tmp_path: Path, mock_config: Config) -> None:
    """Test run method."""
    manager = WorkflowManager(mock_config, state_path=tmp_path / "state.json")
    manager.run()
    assert manager.state.current_phase == WorkflowPhase.EXPLORATION
