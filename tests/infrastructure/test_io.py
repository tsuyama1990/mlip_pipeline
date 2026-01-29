from pathlib import Path
import pytest
import yaml
from mlip_autopipec.infrastructure import io
from mlip_autopipec.domain_models.workflow import WorkflowState, WorkflowPhase

def test_yaml_io(tmp_path: Path) -> None:
    data = {"key": "value", "nested": {"a": 1}}
    path = tmp_path / "test.yaml"

    io.dump_yaml(data, path)
    loaded = io.load_yaml(path)

    assert loaded == data

def test_state_io(tmp_path: Path) -> None:
    state = WorkflowState(cycle_index=1, current_phase=WorkflowPhase.TRAINING)
    path = tmp_path / "state.json"

    io.save_state(state, path)
    loaded_state = io.load_state(path)

    assert loaded_state.cycle_index == 1
    assert loaded_state.current_phase == WorkflowPhase.TRAINING

def test_load_yaml_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        io.load_yaml(tmp_path / "missing.yaml")

def test_load_yaml_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.touch()
    assert io.load_yaml(p) == {}

def test_load_yaml_invalid_type(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    with p.open("w") as f:
        yaml.dump([1, 2], f)
    with pytest.raises(TypeError):
        io.load_yaml(p)

def test_load_state_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        io.load_state(tmp_path / "missing.json")
