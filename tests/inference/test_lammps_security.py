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

def test_lammps_runner_security(basic_config: InferenceConfig, mock_atoms: Atoms, tmp_path: Path) -> None:
    runner = LammpsRunner(basic_config, work_dir=tmp_path)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        result = runner.run(mock_atoms)

        assert result.succeeded is True
        args, kwargs = mock_run.call_args

        # Verify shell=False (default) or not present
        assert kwargs.get("shell") is not True

        # Verify command is a list
        cmd = args[0]
        assert isinstance(cmd, list)
