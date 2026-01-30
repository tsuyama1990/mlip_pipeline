import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import ase
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


@patch("subprocess.run")
def test_run_subprocess_success(mock_run: MagicMock) -> None:
    """Test successful subprocess execution."""
    mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
    stdout, stderr = io.run_subprocess(["echo", "hello"], cwd=Path("/tmp"))
    assert stdout == "ok"
    mock_run.assert_called_with(
        ["echo", "hello"],
        cwd=Path("/tmp"),
        capture_output=True,
        text=True,
        timeout=None,
        check=True,
    )


@patch("subprocess.run")
def test_run_subprocess_timeout(mock_run: MagicMock) -> None:
    """Test subprocess timeout."""
    mock_run.side_effect = subprocess.TimeoutExpired(["cmd"], 1.0)
    with pytest.raises(subprocess.TimeoutExpired):
        io.run_subprocess(["sleep", "2"], cwd=Path("/tmp"), timeout=1.0)


def test_write_lammps_data(tmp_path: Path, sample_ase_atoms: ase.Atoms) -> None:
    """Test writing LAMMPS data file."""
    s = Structure.from_ase(sample_ase_atoms)
    p = tmp_path / "data.lammps"
    io.write_lammps_data(s, p)
    assert p.exists()
    content = p.read_text()
    assert "atoms" in content
    assert "xlo xhi" in content
