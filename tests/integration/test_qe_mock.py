import subprocess

import pytest
from ase import Atoms

from mlip_autopipec.core.config import DFTConfig
from mlip_autopipec.dft.qe_runner import QERunner
from mlip_autopipec.exceptions import DFTCalculationException


@pytest.fixture
def mock_dft_config(tmp_path):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    # Create fake UPF files
    (pseudo_dir / "Si.upf").touch()

    return DFTConfig(
        code="quantum_espresso",
        command="mock_pw.x",
        pseudopotential_dir=pseudo_dir,
    )

def test_run_static_calculation_success(mock_dft_config, tmp_path, mocker):
    runner = QERunner(mock_dft_config)
    atoms = Atoms("Si", cell=[5,5,5], pbc=True)

    # Mock subprocess.run
    mock_run = mocker.patch("subprocess.run")

    # Mock parsing to avoid needing real QE output
    mock_parser = mocker.patch("mlip_autopipec.dft.qe_runner.parse_pw_output")
    from mlip_autopipec.data_models.dft_models import DFTResult
    mock_result = DFTResult(
        uid="test", energy=-1360.5, forces=[[0.0, 0.0, 0.0]], stress=[[0.0]*3]*3,
        succeeded=True, wall_time=1.0, parameters={}, final_mixing_beta=0.7
    )
    mock_parser.return_value = mock_result

    result = runner.run_static_calculation(atoms, tmp_path / "run")

    assert result.energy == -1360.5
    assert mock_run.called
    # Check command structure: ['mock_pw.x', '-in', 'pw.in']
    cmd_args = mock_run.call_args[0][0]
    assert cmd_args[0] == "mock_pw.x"
    assert cmd_args[1] == "-in"

def test_run_static_calculation_failure(mock_dft_config, tmp_path, mocker):
    runner = QERunner(mock_dft_config)
    atoms = Atoms("Si", cell=[5,5,5], pbc=True)

    mock_run = mocker.patch("subprocess.run")
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr=b"Error!")

    with pytest.raises(DFTCalculationException):
        runner.run_static_calculation(atoms, tmp_path / "run_fail")
