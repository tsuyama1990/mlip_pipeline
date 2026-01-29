from pathlib import Path

import pytest
import yaml

from mlip_autopipec.infrastructure import io


def test_load_yaml_valid(tmp_path: Path) -> None:
    """Test loading valid YAML."""
    p = tmp_path / "test.yaml"
    data = {"foo": "bar", "baz": 123}
    with p.open("w") as f:
        yaml.dump(data, f)

    loaded = io.load_yaml(p)
    assert loaded == data


def test_load_yaml_missing() -> None:
    """Test loading missing file."""
    with pytest.raises(FileNotFoundError):
        io.load_yaml(Path("non_existent.yaml"))


def test_dump_yaml(tmp_path: Path) -> None:
    """Test dumping YAML."""
    p = tmp_path / "out.yaml"
    data = {"a": 1, "b": [2, 3]}

    io.dump_yaml(data, p)

    assert p.exists()
    with p.open() as f:
        loaded = yaml.safe_load(f)
    assert loaded == data


def test_load_yaml_empty(tmp_path: Path) -> None:
    """Test loading empty YAML file."""
    p = tmp_path / "empty.yaml"
    p.touch()
    assert io.load_yaml(p) == {}


def test_load_yaml_invalid_type(tmp_path: Path) -> None:
    """Test loading YAML that is not a dict."""
    p = tmp_path / "list.yaml"
    with p.open("w") as f:
        yaml.dump([1, 2, 3], f)

    with pytest.raises(TypeError, match="must contain a dictionary"):
        io.load_yaml(p)
