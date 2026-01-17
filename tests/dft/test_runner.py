from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTErrorType, DFTResult
from mlip_autopipec.dft.runner import QERunner


@pytest.fixture
def mock_dft_config():
    return DFTConfig(
        command="pw.x", pseudo_dir=Path("/tmp/pseudos"), timeout=10, recoverable=True, max_retries=2
    )


@pytest.fixture
def mock_atoms():
    return Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)


@patch("subprocess.run")
@patch("mlip_autopipec.dft.runner.InputGenerator.create_input_string")
@patch("mlip_autopipec.dft.runner.read")
def test_runner_success(mock_read, mock_create_input, mock_subprocess, mock_dft_config, mock_atoms):
    # Setup mocks
    mock_create_input.return_value = "CONTROL..."

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "JOB DONE"
    mock_proc.stderr = ""
    mock_subprocess.return_value = mock_proc

    # Mock ASE read to return an atoms object with calculated results
    result_atoms = mock_atoms.copy()

    # Mock get_potential_energy, get_forces, get_stress
    import numpy as np
    from ase.calculators.singlepoint import SinglePointCalculator

    calc = SinglePointCalculator(
        result_atoms,
        energy=-100.0,
        forces=np.zeros((len(result_atoms), 3)),
        stress=np.zeros(6),  # SinglePointCalculator usually takes Voigt for stress if 6-vector
    )
    # Patch get_stress to return 3x3
    # Actually SinglePointCalculator handles it if we pass 3x3?
    # Let's try to mock get_stress method on the atoms object wrapper if needed,
    # but QERunner calls atoms.get_stress(voigt=False).
    # SinglePointCalculator.get_stress calls self.results.get('stress') and reshapes if needed?
    # ASE doc says SinglePointCalculator stores what is given.

    # If we give 6 components, atoms.get_stress(voigt=False) converts it to 3x3.
    result_atoms.calc = calc

    mock_read.return_value = result_atoms

    runner = QERunner(mock_dft_config)
    result = runner.run(mock_atoms, uid="test-run")

    assert isinstance(result, DFTResult)
    assert result.succeeded is True
    assert result.energy == -100.0
    assert mock_subprocess.call_count == 1


@patch("subprocess.run")
@patch("mlip_autopipec.dft.runner.InputGenerator.create_input_string")
@patch("mlip_autopipec.dft.recovery.RecoveryHandler.analyze")
@patch("mlip_autopipec.dft.recovery.RecoveryHandler.get_strategy")
def test_runner_retry(
    mock_get_strategy, mock_analyze, mock_create_input, mock_subprocess, mock_dft_config, mock_atoms
):
    runner = QERunner(mock_dft_config)

    # First attempt fails
    proc_fail = MagicMock()
    proc_fail.returncode = 1
    proc_fail.stdout = "Error"

    # Second attempt succeeds
    proc_success = MagicMock()
    proc_success.returncode = 0
    proc_success.stdout = "Done"

    mock_subprocess.side_effect = [proc_fail, proc_success]

    mock_analyze.return_value = DFTErrorType.CONVERGENCE_FAIL
    mock_get_strategy.return_value = {"mixing_beta": 0.3}

    # Mock create_input_string to return string, NOT MagicMock
    mock_create_input.return_value = "mock_input_file_content"

    # We need to mock ase.io.read for the success case?
    # Or just assume QERunner handles the second success.
    # For this test, let's assume we mock _parse_result or similar if we extracted it,
    # but here we are testing the loop.

    # Wait, if I mock `run` method it defeats the purpose.
    # I should mock `_run_command` or similar?
    # But I am testing `run`.

    # IMPORTANT: The _parse_output must FAIL on the first attempt so that retry logic is triggered.
    # QERunner logic:
    # 1. run subprocess
    # 2. try parse output -> if success, return result. if fail, proceed to recovery analysis.

    # In my test, first attempt has proc.returncode=1.
    # If parse succeeds anyway (e.g. partial output), it might return?
    # But QERunner does:
    # try:
    #    result = self._parse_output(...)
    #    if result.succeeded: return result
    # except: pass

    # So we need _parse_output to raise exception on first call (or return failed result),
    # and return success on second call.

    with patch("mlip_autopipec.dft.runner.QERunner._parse_output") as mock_parse:
        # First call raises Exception (parse failed because job failed)
        # Second call returns valid result
        mock_parse.side_effect = [
            Exception("Parse failed"),
            DFTResult(
                uid="test-run",
                energy=-100,
                forces=[[0, 0, 0]],
                stress=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                succeeded=True,
                wall_time=10,
                parameters={},
                final_mixing_beta=0.3,
            ),
        ]

        result = runner.run(mock_atoms, uid="test-run")

        assert result.succeeded is True
        assert mock_subprocess.call_count == 2
        assert result.final_mixing_beta == 0.3
