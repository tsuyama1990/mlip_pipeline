from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from ase import Atoms

from mlip_autopipec.core.config import DFTConfig
from mlip_autopipec.core.models import DFTResult
from mlip_autopipec.dft.qe_runner import QERunner


def test_qe_runner_mock(tmp_path: Path, mocker: MagicMock) -> None:
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

    run_dir = tmp_path / "run"
    atoms = Atoms("Si", cell=[5,5,5], pbc=True)

    # Mock subprocess.run
    mock_run = mocker.patch("subprocess.run")

    def side_effect(*args, **kwargs):
        # Write dummy output to stdout file handle
        # This ensures 'pw.out' is created, preventing FileNotFoundError in runner
        stdout_handle = kwargs.get("stdout")
        if stdout_handle:
            stdout_handle.write("JOB DONE.\n")
            stdout_handle.flush()
        return MagicMock(returncode=0)

    mock_run.side_effect = side_effect

    # Mock parse_pw_output to return a predefined result
    # This avoids needing to write valid QE output for ASE to parse
    expected_result = DFTResult(
        energy=-100.0,
        forces=np.zeros((1,3)),
        stress=np.zeros((3,3)) # Voigt (6) or Matrix (3x3)? ASE stress is 6 (Voigt) or 3x3.
        # DFTResult model expects Array.
        # ase.get_stress() returns array of 6 elements (Voigt form) by default.
        # But let's check DFTResult model again. "stress: Optional[NDArray]".
        # I'll use 6 elements.
    )
    # Note: validation of shape is not strictly enforced by NDArray without validator, but good to match ASE.
    expected_result.stress = np.zeros(6)

    mocker.patch("mlip_autopipec.dft.qe_runner.parse_pw_output", return_value=expected_result)

    result = runner.run_static_calculation(atoms, run_dir)

    assert result == expected_result
    assert (run_dir / "pw.in").exists()
    mock_run.assert_called_once()
