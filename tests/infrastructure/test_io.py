from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch

import ase
import numpy as np
import pytest
import yaml

from mlip_autopipec.domain_models.structure import Structure
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

def test_run_subprocess_success() -> None:
    """Test successful subprocess execution."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="ok", stderr="", returncode=0)

        stdout, stderr = io.run_subprocess(["ls", "-l"])

        assert stdout == "ok"
        mock_run.assert_called_once()
        # Verify args: check that args[0] is the command list
        assert mock_run.call_args[0][0] == ["ls", "-l"]

def test_run_subprocess_failure() -> None:
    """Test failed subprocess execution."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, ["cmd"], stderr="err")

        # Should raise the original error unless we wrap it.
        # Spec implies checking status.
        # But run_subprocess helper usually raises or returns.
        # If I want to catch it in LammpsRunner, I should let it raise here.
        with pytest.raises(subprocess.CalledProcessError):
            io.run_subprocess(["cmd"])

def test_run_subprocess_timeout() -> None:
    """Test subprocess timeout."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(["cmd"], 1.0)

        # Spec says: "Assert LammpsRunner returns a result with status=TIMEOUT".
        # So io.run_subprocess should probably raise TimeoutExpired (wrapped as TimeoutError?).
        # I'll raise TimeoutError for now.
        with pytest.raises(TimeoutError):
            io.run_subprocess(["cmd"], timeout=1.0)

def test_write_lammps_data(tmp_path: Path) -> None:
    """Test writing LAMMPS data."""
    s = Structure(
        symbols=["Si"],
        positions=np.array([[0,0,0]]),
        cell=np.eye(3)*5,
        pbc=(True,True,True)
    )
    p = tmp_path / "data.lammps"
    io.write_lammps_data(s, p)

    assert p.exists()
    content = p.read_text()
    assert "Atoms" in content
    assert "Si" in content or "1" in content # atom type

def test_read_lammps_dump_mocked(tmp_path: Path) -> None:
    """Test reading LAMMPS dump using mock ASE."""
    p = tmp_path / "dump.lammpstrj"
    p.touch()

    with patch("ase.io.read") as mock_read:
        atoms = ase.Atoms("H", positions=[[0,0,0]], cell=[10,10,10], pbc=True)
        # ase.io.read returns Atoms object or list of Atoms.
        # We assume it returns the last frame (Atoms) if index is -1.
        mock_read.return_value = atoms

        s = io.read_lammps_dump(p)
        assert s.symbols == ["H"]
