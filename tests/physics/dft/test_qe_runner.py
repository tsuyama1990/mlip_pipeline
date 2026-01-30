from unittest.mock import patch
import pytest
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.calculation import DFTConfig, DFTResult, SCFError
from mlip_autopipec.physics.dft.qe_runner import QERunner
import numpy as np

@pytest.fixture
def mock_structure():
    return Structure(
        symbols=["Si"],
        positions=np.zeros((1, 3)),
        cell=np.eye(3),
        pbc=(True, True, True)
    )

@pytest.fixture
def mock_config(tmp_path):
    # Create dummy UPF
    upf_path = tmp_path / "Si.UPF"
    upf_path.touch()
    return DFTConfig(
        command="pw.x",
        pseudopotentials={"Si": upf_path},
        ecutwfc=30.0,
        kspacing=0.04
    )

@patch("mlip_autopipec.physics.dft.qe_runner.subprocess.run")
@patch("mlip_autopipec.physics.dft.qe_runner.InputGenerator")
@patch("mlip_autopipec.physics.dft.qe_runner.Parser")
def test_run_success(MockParser, MockInputGen, mock_run, mock_structure, mock_config, tmp_path):
    # Setup mocks
    MockInputGen.generate.return_value = "control..."
    mock_run.return_value.stdout = "Job Done"
    mock_run.return_value.stderr = ""

    expected_result = DFTResult(
        energy=-10.0,
        forces=np.zeros((1, 3)),
        stress=np.zeros((3, 3))
    )
    MockParser.parse.return_value = expected_result

    runner = QERunner(working_dir=tmp_path / "work")
    result = runner.run(mock_structure, mock_config)

    assert result == expected_result
    mock_run.assert_called_once()
    MockInputGen.generate.assert_called_with(mock_structure, mock_config)

@patch("mlip_autopipec.physics.dft.qe_runner.subprocess.run")
@patch("mlip_autopipec.physics.dft.qe_runner.InputGenerator")
@patch("mlip_autopipec.physics.dft.qe_runner.Parser")
@patch("mlip_autopipec.physics.dft.qe_runner.RecoveryHandler")
def test_run_recovery(MockRecovery, MockParser, MockInputGen, mock_run, mock_structure, mock_config, tmp_path):
    # Setup mocks
    MockInputGen.generate.return_value = "control..."

    # First run fails with SCFError
    # Second run succeeds

    # Mock Parser to raise error then return result
    expected_result = DFTResult(
        energy=-10.0,
        forces=np.zeros((1, 3)),
        stress=np.zeros((3, 3))
    )

    MockParser.parse.side_effect = [SCFError("Fail"), expected_result]

    # Mock Recovery to return modified config
    modified_config = mock_config.model_copy(update={"mixing_beta": 0.3})
    MockRecovery.return_value.apply_fix.return_value = modified_config
    MockRecovery.return_value.MAX_ATTEMPTS = 5

    runner = QERunner(working_dir=tmp_path / "work")
    result = runner.run(mock_structure, mock_config)

    assert result == expected_result
    assert mock_run.call_count == 2
    MockRecovery.return_value.apply_fix.assert_called_once()
