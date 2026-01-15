# ruff: noqa: D101, D102, D103
"""Tests for the QEProcessRunner."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest
from ase.build import bulk

from mlip_autopipec.config_schemas import SystemConfig
from mlip_autopipec.modules.dft.dft_factory import (
    DFTCalculationError,
    QEInputGenerator,
    QEOutputParser,
    QEProcessRunner,
)


@pytest.fixture
def mock_config() -> MagicMock:
    """Return a properly structured mock for SystemConfig."""
    mock = MagicMock(spec=SystemConfig)

    # Create nested mocks for the DFT config structure
    mock.dft = MagicMock()
    mock.dft.input = MagicMock()
    mock.dft.executable = MagicMock()

    # Set the return value for the model_copy call
    mock.dft.input.model_copy.return_value = MagicMock()

    # Set the retry and executable parameters
    mock.dft.max_retries = 2
    mock.dft.retry_strategy = [
        {"params": {"electrons": {"mixing_beta": 0.5}}},
        {"params": {"electrons": {"mixing_beta": 0.3}}},
    ]
    mock.dft.executable.command = "pw.x"

    return mock


@pytest.fixture
def qe_process_runner(mock_config: SystemConfig) -> QEProcessRunner:
    """Return an instance of the QEProcessRunner."""
    mock_config.dft.max_retries = 2
    mock_config.dft.retry_strategy = [
        {"params": {"electrons": {"mixing_beta": 0.5}}},
        {"params": {"electrons": {"mixing_beta": 0.3}}},
    ]
    input_generator = QEInputGenerator()
    output_parser = QEOutputParser()
    return QEProcessRunner(mock_config, input_generator, output_parser)


def test_qe_runner_successful_first_attempt(
    qe_process_runner: QEProcessRunner,
) -> None:
    """Test that the runner succeeds on the first try."""
    atoms = bulk("Si", "diamond", a=5.43)
    with patch("subprocess.run") as mock_subprocess, patch(
        "mlip_autopipec.modules.dft.dft_factory.QEOutputParser.parse_output"
    ) as mock_parse, patch("shutil.which", return_value="pw.x"):
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_parse.return_value = atoms
        result_atoms = qe_process_runner.run(atoms)
        assert result_atoms is not None
        mock_subprocess.assert_called_once()


def test_qe_runner_retry_logic(qe_process_runner: QEProcessRunner) -> None:
    """Test the retry logic of the QEProcessRunner."""
    atoms = bulk("Si", "diamond", a=5.43)
    with patch("subprocess.run") as mock_subprocess, patch(
        "mlip_autopipec.modules.dft.dft_factory.QEInputGenerator.generate_input"
    ) as mock_generate_input, patch(
        "mlip_autopipec.modules.dft.dft_factory.QEOutputParser.parse_output"
    ) as mock_parse, patch("shutil.which", return_value="pw.x"):
        # Simulate failure on the first call, success on the second
        mock_subprocess.side_effect = [
            subprocess.CalledProcessError(1, "pw.x", stderr="SCF not converged"),
            MagicMock(returncode=0),
        ]
        mock_parse.return_value = atoms

        qe_process_runner.run(atoms)

        # Assert that subprocess.run was called twice
        assert mock_subprocess.call_count == 2

        # Assert that the input generator was called with modified config on 2nd attempt
        final_call_args = mock_generate_input.call_args_list[1]
        modified_dft_input = final_call_args[0][2] # The dft_input argument
        assert modified_dft_input.electrons.mixing_beta == 0.5


def test_qe_runner_all_attempts_fail(qe_process_runner: QEProcessRunner) -> None:
    """Test that a DFTCalculationError is raised if all retries fail."""
    atoms = bulk("Si", "diamond", a=5.43)
    with patch("subprocess.run") as mock_subprocess, patch(
        "shutil.which", return_value="pw.x"
    ):
        # Simulate failure for all attempts
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            1, "pw.x", stderr="SCF not converged"
        )

        with pytest.raises(DFTCalculationError):
            qe_process_runner.run(atoms)

        # Initial call + 2 retries
        assert mock_subprocess.call_count == qe_process_runner.config.dft.max_retries + 1
