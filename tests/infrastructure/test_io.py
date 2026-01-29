"""Tests for I/O utilities."""

from pathlib import Path

import pytest

from mlip_autopipec.infrastructure import io


def test_yaml_io(tmp_path: Path) -> None:
    """Test YAML load and dump."""
    data = {"key": "value", "nested": {"a": 1}}
    path = tmp_path / "test.yaml"

    io.dump_yaml(data, path)
    assert path.exists()

    loaded = io.load_yaml(path)
    assert loaded == data


def test_load_yaml_not_found() -> None:
    """Test loading non-existent file."""
    with pytest.raises(FileNotFoundError):
        io.load_yaml("non_existent.yaml")


def test_json_io(tmp_path: Path) -> None:
    """Test JSON save and load."""
    data = {"state": "RUNNING", "cycle": 1}
    path = tmp_path / "state.json"

    io.save_json(data, path)
    assert path.exists()

    loaded = io.load_json(path)
    assert loaded == data


def test_load_json_not_found() -> None:
    """Test loading non-existent JSON."""
    with pytest.raises(FileNotFoundError):
        io.load_json("non_existent.json")

def test_load_yaml_empty(tmp_path: Path) -> None:
    """Test loading an empty YAML file."""
    path = tmp_path / "empty.yaml"
    path.touch()
    assert io.load_yaml(path) == {}

def test_load_yaml_invalid(tmp_path: Path) -> None:
    """Test loading invalid YAML."""
    path = tmp_path / "invalid.yaml"
    path.write_text("invalid: [unclosed")

    # yaml.safe_load raises ScannerError or ParserError
    from yaml import YAMLError
    with pytest.raises(YAMLError):
        io.load_yaml(path)

def test_load_json_malformed(tmp_path: Path) -> None:
    """Test loading malformed JSON."""
    path = tmp_path / "malformed.json"
    path.write_text("{unquoted: 1}")

    # json.load raises JSONDecodeError which inherits from ValueError
    # We use a broad match because the message depends on the python version/implementation
    with pytest.raises(ValueError, match="Expecting property name"):
        io.load_json(path)
