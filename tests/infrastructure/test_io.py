from pathlib import Path

import pytest
import yaml

from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState
from mlip_autopipec.infrastructure import io

# Constants
TEST_YAML_FILENAME = "test.yaml"
TEST_STATE_FILENAME = "state.json"


def test_load_yaml_valid(tmp_path: Path) -> None:
    """Test loading a valid YAML file."""
    data = {"key": "value", "nested": {"a": 1}}
    path = tmp_path / TEST_YAML_FILENAME

    # Use safe_dump for test setup
    with path.open("w") as f:
        yaml.safe_dump(data, f)

    loaded = io.load_yaml(path)
    assert loaded == data


def test_dump_yaml(tmp_path: Path) -> None:
    """Test dumping data to a YAML file."""
    data = {"key": "value", "list": [1, 2, 3]}
    path = tmp_path / "output.yaml"

    io.dump_yaml(data, path)

    # Verify with safe_load
    with path.open("r") as f:
        loaded = yaml.safe_load(f)
    assert loaded == data


def test_load_yaml_missing(tmp_path: Path) -> None:
    """Test loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        io.load_yaml(tmp_path / "missing.yaml")


def test_load_yaml_empty(tmp_path: Path) -> None:
    """Test loading an empty file returns an empty dict."""
    p = tmp_path / "empty.yaml"
    p.touch()
    assert io.load_yaml(p) == {}


def test_load_yaml_invalid_syntax(tmp_path: Path) -> None:
    """Test loading invalid YAML syntax raises YAMLError."""
    p = tmp_path / "invalid.yaml"
    # Use tab for indentation which is invalid in YAML
    p.write_text("key:\n\tvalue", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        io.load_yaml(p)


def test_load_yaml_invalid_type(tmp_path: Path) -> None:
    """Test loading a valid YAML that is not a dictionary raises TypeError."""
    p = tmp_path / "list.yaml"
    with p.open("w") as f:
        yaml.safe_dump([1, 2], f)

    # Verify it loads as list first (sanity check for test)
    with p.open("r") as f:
        assert isinstance(yaml.safe_load(f), list)

    # Verify function raises TypeError
    with pytest.raises(TypeError) as excinfo:
        io.load_yaml(p)
    assert "must contain a dictionary" in str(excinfo.value)


def test_state_io(tmp_path: Path) -> None:
    """Test saving and loading WorkflowState."""
    state = WorkflowState(cycle_index=1, current_phase=WorkflowPhase.TRAINING)
    path = tmp_path / TEST_STATE_FILENAME

    io.save_state(state, path)

    assert path.exists()

    loaded_state = io.load_state(path)
    assert loaded_state.cycle_index == 1
    assert loaded_state.current_phase == WorkflowPhase.TRAINING


def test_load_state_missing(tmp_path: Path) -> None:
    """Test loading missing state file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        io.load_state(tmp_path / "missing.json")
