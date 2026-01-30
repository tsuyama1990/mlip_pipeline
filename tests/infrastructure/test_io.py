from pathlib import Path
from unittest.mock import Mock, patch
import subprocess

import pytest
import yaml
import numpy as np

from mlip_autopipec.infrastructure import io
from mlip_autopipec.domain_models.structure import Structure


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

# Cycle 02 Tests

def test_write_lammps_data(tmp_path: Path) -> None:
    """Test writing structure to LAMMPS data format."""
    s = Structure(
        symbols=["Si"],
        positions=np.array([[0, 0, 0]]),
        cell=np.eye(3) * 5.43,
        pbc=(True, True, True)
    )
    p = tmp_path / "data.lammps"
    io.write_lammps_data(s, p)
    assert p.exists()
    content = p.read_text()
    assert "atoms" in content
    assert "xlo xhi" in content

def test_read_lammps_dump(tmp_path: Path) -> None:
    """Test reading structure from LAMMPS dump file."""
    # Write a dummy dump file
    dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
2 1 5.0 5.0 5.0
"""
    p = tmp_path / "dump.lammpstrj"
    p.write_text(dump_content)

    s = io.read_lammps_dump(p, species=["Ar"])
    assert len(s.positions) == 2
    assert s.symbols == ["Ar", "Ar"]

@patch("subprocess.run")
def test_run_subprocess_success(mock_run: Mock) -> None:
    """Test running a subprocess successfully."""
    mock_run.return_value = Mock(returncode=0, stdout="Success", stderr="")

    io.run_subprocess("echo hello", timeout=10)

    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert args[0] == ["echo", "hello"]
    assert kwargs["timeout"] == 10

@patch("subprocess.run")
def test_run_subprocess_failure(mock_run: Mock) -> None:
    """Test running a subprocess with failure."""
    mock_run.side_effect = subprocess.CalledProcessError(1, ["crash"], stderr="Error")

    with pytest.raises(RuntimeError):
        io.run_subprocess("crash", timeout=10)
