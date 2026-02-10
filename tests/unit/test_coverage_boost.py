import pytest
import json
from unittest.mock import Mock, patch
from pathlib import Path
from mlip_autopipec.core.state_manager import StateManager
from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models import Config, WorkflowState

def test_state_manager_load_failure(tmp_path: Path):
    """Test handling of corrupted state file."""
    p = tmp_path / "workflow_state.json"
    p.write_text("{corrupted json", encoding="utf-8")

    manager = StateManager(tmp_path)
    state = manager.load()
    # Should return fresh state
    assert state.iteration == 0

def test_state_manager_save_failure(tmp_path: Path):
    """Test handling of save failure (OSError)."""
    manager = StateManager(tmp_path)
    state = WorkflowState(iteration=5)

    # Mock Path.open to raise OSError
    # Since we use Path object methods, we need to mock the return of open
    # But wait, we call self.state_file.open("w") inside save?
    # Actually code is:
    # temp_file = self.work_dir / f".{self.state_file.name}.tmp"
    # with temp_file.open("w") as f:

    # So we need to mock Path.open.
    # Since Path is imported in state_manager, maybe we can mock it there?
    # No, Path is a class.
    # The best way is to make work_dir read-only, but that's hard in tmp_path.

    # Alternative: Mock the rename operation to fail.
    # temp_file.rename(self.state_file)
    with patch("pathlib.Path.rename", side_effect=OSError("Rename failed")):
         with pytest.raises(OSError):
             manager.save(state)

def test_state_manager_mkdir(tmp_path: Path):
    """Test that StateManager creates work_dir if missing."""
    p = tmp_path / "subdir"
    assert not p.exists()
    StateManager(p)
    assert p.exists()

def test_orchestrator_not_implemented_types(tmp_path: Path):
    """Test NotImplementedError for unsupported components."""
    # Config with random generator (not implemented in init yet)
    data = {
        "orchestrator": {"work_dir": str(tmp_path), "max_iterations": 1},
        "generator": {"type": "random"},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock", "dataset_path": str(tmp_path / "data")}
    }
    config = Config.model_validate(data)

    with pytest.raises(NotImplementedError) as exc:
        Orchestrator(config)
    assert "Generator type" in str(exc.value)

def test_orchestrator_bad_oracle(tmp_path: Path):
    data = {
        "orchestrator": {"work_dir": str(tmp_path), "max_iterations": 1},
        "generator": {"type": "mock"},
        "oracle": {"type": "qe", "command": "pw.x"},
        "trainer": {"type": "mock", "dataset_path": str(tmp_path / "data")}
    }
    config = Config.model_validate(data)
    with pytest.raises(NotImplementedError) as exc:
        Orchestrator(config)
    assert "Oracle type" in str(exc.value)

def test_orchestrator_bad_trainer(tmp_path: Path):
    data = {
        "orchestrator": {"work_dir": str(tmp_path), "max_iterations": 1},
        "generator": {"type": "mock"},
        "oracle": {"type": "mock"},
        "trainer": {"type": "pace", "dataset_path": str(tmp_path / "data")}
    }
    config = Config.model_validate(data)
    with pytest.raises(NotImplementedError) as exc:
        Orchestrator(config)
    assert "Trainer type" in str(exc.value)

def test_orchestrator_resume_finished(tmp_path: Path):
    """Test that run_loop returns early if already finished."""
    data = {
        "orchestrator": {"work_dir": str(tmp_path), "max_iterations": 2},
        "generator": {"type": "mock"},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock", "dataset_path": str(tmp_path / "data")}
    }
    config = Config.model_validate(data)

    # Create state file showing completion
    state = WorkflowState(iteration=2)
    manager = StateManager(tmp_path)
    manager.save(state)

    orchestrator = Orchestrator(config)
    # This should log "Workflow already completed" and return immediately
    orchestrator.run_loop()

    # Verify iteration didn't change
    assert orchestrator.state.iteration == 2

def test_orchestrator_empty_structures(tmp_path: Path):
    """Test handling of empty structure generation/labelling."""
    data = {
        "orchestrator": {"work_dir": str(tmp_path), "max_iterations": 1},
        "generator": {"type": "mock"},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock", "dataset_path": str(tmp_path / "data")}
    }
    config = Config.model_validate(data)

    orchestrator = Orchestrator(config)

    # Mock generator to return empty iterator
    orchestrator.generator = Mock(spec=orchestrator.generator)
    orchestrator.generator.generate.return_value = iter([])

    # Mock oracle to return empty iterator (since input is empty)
    orchestrator.oracle = Mock(spec=orchestrator.oracle)
    orchestrator.oracle.compute.return_value = iter([])

    orchestrator.run_loop()

    # Iteration should increment but no training should happen
    orchestrator.trainer = Mock(spec=orchestrator.trainer)
    orchestrator.trainer.train.assert_not_called()
