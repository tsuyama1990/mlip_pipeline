"""
Unit tests for the refactored DFTFactory and its dependencies.
"""
import logging
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from ase.build import bulk

from mlip_autopipec.config.models import DFTInputParameters
from mlip_autopipec.exceptions import DFTCalculationError
from mlip_autopipec.modules.dft import (
    DFTFactory,
    QEInputGenerator,
    QEProcessRunner,
)


@pytest.fixture
def mock_input_generator():
    return MagicMock(spec=QEInputGenerator)


@pytest.fixture
def mock_process_runner():
    return MagicMock(spec=QEProcessRunner)


@pytest.fixture
def dft_factory(mock_input_generator, mock_process_runner):
    """Fixture to create a DFTFactory with mocked dependencies."""
    return DFTFactory(
        input_generator=mock_input_generator,
        process_runner=mock_process_runner,
    )


def test_dft_factory_run_successful(
    dft_factory, mock_input_generator, mock_process_runner
):
    """Test a straightforward, successful run."""
    atoms = bulk("Si")
    mock_ase_read = MagicMock()
    mock_ase_read.return_value.info = {
        "energy": -100.0,
        "forces": [[0.0] * 3] * 2,
        "stress": [0.0] * 6,
    }

    with patch("mlip_autopipec.modules.dft.ase_read", mock_ase_read):
        result = dft_factory.run(atoms)

    mock_input_generator.prepare_input_files.assert_called_once()
    mock_process_runner.execute.assert_called_once()
    assert result.energy == -100.0


def test_dft_factory_retry_and_succeed(
    dft_factory, mock_input_generator, mock_process_runner
):
    """Test the retry mechanism where a run fails once, then succeeds."""
    atoms = bulk("Si")
    # Correctly create the exception object
    error = subprocess.CalledProcessError(1, "pw.x")
    error.stdout = "convergence NOT achieved"
    error.stderr = ""
    mock_process_runner.execute.side_effect = [error, MagicMock()]

    mock_ase_read = MagicMock()
    mock_ase_read.return_value.info = {
        "energy": -150.0,
        "forces": [[0.0] * 3] * 2,
        "stress": [0.0] * 6,
    }

    with patch("mlip_autopipec.modules.dft.ase_read", mock_ase_read):
        result = dft_factory.run(atoms)

    assert mock_process_runner.execute.call_count == 2
    assert result.energy == -150.0


def test_dft_factory_fails_after_max_retries(
    dft_factory, mock_input_generator, mock_process_runner
):
    """Test that the factory gives up after the maximum number of retries."""
    atoms = bulk("Si")
    error = subprocess.CalledProcessError(1, "pw.x")
    error.stdout = "convergence NOT achieved"
    error.stderr = ""
    mock_process_runner.execute.side_effect = [error, error, error]

    with pytest.raises(DFTCalculationError):
        dft_factory.run(atoms)

    assert mock_process_runner.execute.call_count == 3


def test_handle_convergence_error():
    """Unit test for the convergence error handling logic."""
    factory = DFTFactory(MagicMock(), MagicMock())
    params = {"mixing_beta": 0.7}
    log = "convergence NOT achieved"
    new_params = factory._handle_convergence_error(log, params)
    assert new_params["mixing_beta"] == 0.35

    params = {"diagonalization": "david"}
    log = "Cholesky"
    new_params = factory._handle_convergence_error(log, params)
    assert new_params["diagonalization"] == "cg"

    params = {}
    log = "some other error"
    new_params = factory._handle_convergence_error(log, params)
    assert new_params is None


def test_process_runner_logs_on_error(caplog, tmp_path):
    """Verify that the QEProcessRunner logs stdout/stderr on failure."""
    mock_profile = MagicMock()
    mock_profile.get_command.return_value = ["echo", "fail"]
    runner = QEProcessRunner(profile=mock_profile)

    error = subprocess.CalledProcessError(returncode=1, cmd="pw.x")
    error.stdout = "Test stdout"
    error.stderr = "Test stderr"

    input_file = tmp_path / "test.in"
    output_file = tmp_path / "test.out"
    input_file.touch()

    with patch("subprocess.run", side_effect=error):
        with pytest.raises(subprocess.CalledProcessError):
            runner.execute(input_file, output_file)

    assert "QE subprocess failed" in caplog.text
    assert "Test stdout" in caplog.text
    assert "Test stderr" in caplog.text
