from pathlib import Path
from unittest.mock import patch

import pytest
from ase.atoms import Atoms

from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.inference.lammps_runner import LammpsRunner


@pytest.fixture
def mock_atoms() -> Atoms:
    return Atoms("Al4", pbc=True, cell=[4,4,4])

@pytest.fixture
def basic_config(tmp_path: Path) -> InferenceConfig:
    p = tmp_path / "model.yace"
    p.touch()
    l = tmp_path / "lmp_serial"
    l.touch(mode=0o755)
    return InferenceConfig(
        temperature=300.0,
        potential_path=p,
        lammps_executable=l
    )

def test_lammps_runner_execution_success(basic_config: InferenceConfig, mock_atoms: Atoms, tmp_path: Path) -> None:
    runner = LammpsRunner(basic_config, work_dir=tmp_path)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "LAMMPS output"

        result = runner.run(mock_atoms)

        assert result.succeeded is True
        assert (tmp_path / "in.lammps").exists()
        assert (tmp_path / "data.lammps").exists()

        # Check if subprocess was called correctly
        # We need to type ignore call_args because mock
        args, kwargs = mock_run.call_args
        cmd = args[0]
        # lammps_executable is Optional, but basic_config sets it.
        assert basic_config.lammps_executable is not None
        assert cmd[0] == str(basic_config.lammps_executable)
        assert "-in" in cmd
        assert str(tmp_path / "in.lammps") in cmd

def test_lammps_runner_failure(basic_config: InferenceConfig, mock_atoms: Atoms, tmp_path: Path) -> None:
    runner = LammpsRunner(basic_config, work_dir=tmp_path)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1 # Fail
        mock_run.return_value.stdout = "Error"

        result = runner.run(mock_atoms)
        assert result.succeeded is False
