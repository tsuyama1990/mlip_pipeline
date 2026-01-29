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
