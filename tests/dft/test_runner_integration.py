from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTErrorType
from mlip_autopipec.dft.runner import DFTFatalError, QERunner


def test_runner_propagates_fatal_exceptions():
    """Test that fatal exceptions during execution are propagated."""
    config = DFTConfig(command="pw.x", pseudo_dir=Path("/tmp"), max_retries=0)
    runner = QERunner(config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True) # Must have cell for input generator

    with patch("subprocess.run") as mock_run:
        # Simulate immediate failure that is not recoverable
        mock_run.side_effect = Exception("System Crash")

        with pytest.raises(Exception) as excinfo:
            runner.run(atoms)
        assert "System Crash" in str(excinfo.value)

def test_runner_raises_after_retries():
    """Test that DFTFatalError is raised after retries executed."""
    config = DFTConfig(command="pw.x", pseudo_dir=Path("/tmp"), max_retries=1)
    runner = QERunner(config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    with patch("subprocess.run") as mock_run, \
         patch("mlip_autopipec.dft.inputs.InputGenerator.create_input_string") as mock_input, \
         patch("mlip_autopipec.dft.recovery.RecoveryHandler.analyze", return_value=DFTErrorType.CONVERGENCE_FAIL), \
         patch("mlip_autopipec.dft.recovery.RecoveryHandler.get_strategy", return_value={}):

        mock_input.return_value = "dummy input"
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_run.return_value = mock_proc

        with pytest.raises(DFTFatalError):
            runner.run(atoms)
