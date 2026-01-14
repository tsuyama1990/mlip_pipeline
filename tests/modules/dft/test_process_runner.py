"""Unit tests for the QEProcessRunner."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.config_schemas import SystemConfig
from mlip_autopipec.modules.dft.exceptions import DFTCalculationError
from mlip_autopipec.modules.dft.process_runner import QEProcessRunner


@patch("subprocess.run")
def test_qeprocessrunner_execute_failure(
    mock_subprocess_run: MagicMock, sample_system_config: SystemConfig
) -> None:
    """Test that QEProcessRunner raises an error on subprocess failure."""
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="pw.x", stderr="SCF failed to converge"
    )

    runner = QEProcessRunner(sample_system_config.dft.executable)
    with pytest.raises(DFTCalculationError, match="DFT calculation failed"):
        runner.execute(Path("dummy.in"), Path("dummy.out"))
