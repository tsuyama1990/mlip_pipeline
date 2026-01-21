import pytest
from unittest.mock import MagicMock
from pathlib import Path
from ase import Atoms
import subprocess
from mlip_autopipec.dft.qe_runner import QERunner
from mlip_autopipec.core.config import DFTConfig
from mlip_autopipec.core.exceptions import DFTRuntimeError

def test_qe_runner_timeout(tmp_path: Path, mocker: MagicMock) -> None:
    # Setup
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    (pseudo_dir / "Si.upf").touch()

    config = DFTConfig(
        command="pw.x",
        pseudopotential_dir=pseudo_dir,
        ecutwfc=30
    )
    runner = QERunner(config)
    atoms = Atoms("Si", cell=[5,5,5], pbc=True)
    run_dir = tmp_path / "run"

    # Mock subprocess.run to raise TimeoutExpired
    mock_run = mocker.patch("subprocess.run")
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="pw.x", timeout=14400)

    with pytest.raises(DFTRuntimeError, match="DFT calculation timed out"):
        runner.run_static_calculation(atoms, run_dir)
