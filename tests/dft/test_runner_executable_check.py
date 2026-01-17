import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from ase import Atoms
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.dft.runner import QERunner, DFTFatalError

def test_runner_executable_missing():
    config = DFTConfig(command="non_existent_executable", pseudo_dir=Path("/tmp"))
    runner = QERunner(config)
    atoms = Atoms("H")

    with patch("shutil.which", return_value=None):
        with pytest.raises(DFTFatalError) as excinfo:
            runner.run(atoms)
        assert "Executable 'non_existent_executable' not found" in str(excinfo.value)

def test_runner_executable_found():
    config = DFTConfig(command="pw.x", pseudo_dir=Path("/tmp"))
    runner = QERunner(config)
    atoms = Atoms("H")

    with patch("shutil.which", return_value="/usr/bin/pw.x"),          patch("subprocess.run") as mock_run,          patch("mlip_autopipec.dft.inputs.InputGenerator.create_input_string", return_value=""),          patch("mlip_autopipec.dft.runner.QERunner._parse_output") as mock_parse:

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        # Mock successful parsing
        mock_parse.return_value.succeeded = True

        # Should not raise
        try:
            runner.run(atoms)
        except DFTFatalError:
            pytest.fail("DFTFatalError raised despite executable existing")
