"""Tests for I/O utilities."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from mlip_autopipec.infrastructure import io


class MockModel(BaseModel):
    key: str
    value: int


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
    with pytest.raises(ValueError, match="Expecting property name"):
        io.load_json(path)

def test_load_yaml_too_large(tmp_path: Path) -> None:
    """Test loading oversized YAML."""
    path = tmp_path / "large.yaml"
    # Create file slightly larger than MAX_CONFIG_SIZE (5MB)
    size = io.MAX_CONFIG_SIZE + 100
    with path.open("wb") as f:
        f.seek(size - 1)
        f.write(b"\0")

    with pytest.raises(ValueError, match="exceeds maximum allowed size"):
        io.load_yaml(path)

def test_load_json_too_large(tmp_path: Path) -> None:
    """Test loading oversized JSON."""
    path = tmp_path / "large.json"
    # Create file slightly larger than MAX_CONFIG_SIZE (5MB)
    size = io.MAX_CONFIG_SIZE + 100
    with path.open("wb") as f:
        f.seek(size - 1)
        f.write(b"\0")

    with pytest.raises(ValueError, match="exceeds maximum allowed size"):
        io.load_json(path)

def test_load_pydantic_from_yaml(tmp_path: Path) -> None:
    """Test loading a Pydantic model from YAML."""
    data = {"key": "test", "value": 42}
    path = tmp_path / "model.yaml"
    io.dump_yaml(data, path)

    model = io.load_pydantic_from_yaml(path, MockModel)
    assert model.key == "test"
    assert model.value == 42
