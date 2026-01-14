from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from mlip_autopipec.modules.c_dft_factory import QERunner
from mlip_autopipec.schemas.dft import DFTInput, DFTOutput
from mlip_autopipec.schemas.system_config import DFTParams


@pytest.fixture
def mock_dft_input() -> DFTInput:
    """Provides a mock DFTInput object for testing."""
    atoms = Atoms("H", positions=[(0, 0, 0)])
    dft_params = DFTParams(
        pseudopotentials={"H": "H.usp"},
        cutoff_wfc=40,
        k_points=(1, 1, 1),
        smearing_type="gauss",
        degauss=0.01,
        nspin=1,
    )
    return DFTInput(atoms=atoms, dft_params=dft_params)


def test_qe_runner_successful_run(mock_dft_input: DFTInput) -> None:
    """
    Test a successful Quantum Espresso run.
    """
    runner = QERunner()
    fake_qe_output = """
    !    total energy              =     -123.456 Ry
    """

    # Mock the subprocess call to simulate QE running successfully
    with patch("subprocess.run") as mock_subprocess:
        # Configure the mock to return a successful process with fake output
        mock_subprocess.return_value = MagicMock(
            returncode=0, stdout=fake_qe_output, stderr=""
        )

        output = runner.run(mock_dft_input)

        # Assert that the output is parsed correctly
        assert isinstance(output, DFTOutput)
        assert output.total_energy == pytest.approx(-123.456)
        mock_subprocess.assert_called_once()


import shutil


def test_qe_runner_recovers_from_convergence_error(mock_dft_input: DFTInput) -> None:
    """
    Test that the QERunner can recover from a convergence error.
    """
    runner = QERunner(max_retries=2, keep_temp_dir=True)
    failed_output = "convergence NOT achieved"
    successful_output = "!    total energy              =     -123.456 Ry"

    with patch("subprocess.run") as mock_subprocess:
        # Simulate one failure, then one success
        mock_subprocess.side_effect = [
            MagicMock(returncode=1, stdout=failed_output, stderr=""),
            MagicMock(returncode=0, stdout=successful_output, stderr=""),
        ]

        output = runner.run(mock_dft_input)
        assert output.total_energy == pytest.approx(-123.456)
        assert mock_subprocess.call_count == 2

        from pathlib import Path

        # Check that the mixing_beta was changed in the second call
        second_call_command = mock_subprocess.call_args_list[1].args[0]
        input_file_path = Path(second_call_command[2])
        with open(input_file_path) as f:
            input_file_content = f.read()
        assert "mixing_beta" in input_file_content
        assert "0.3" in input_file_content
        shutil.rmtree(input_file_path.parent)
