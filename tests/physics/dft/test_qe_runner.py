from unittest.mock import patch
from pathlib import Path
import pytest
import ase
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.calculation import DFTConfig, DFTResult
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.physics.dft.qe_runner import QERunner
import numpy as np

@pytest.fixture
def mock_runner_deps():
    with patch("mlip_autopipec.physics.dft.qe_runner.InputGenerator") as mock_igen, \
         patch("mlip_autopipec.physics.dft.qe_runner.DFTParser") as mock_parser, \
         patch("subprocess.run") as mock_run, \
         patch("pathlib.Path.write_text") as mock_write, \
         patch("pathlib.Path.read_text") as mock_read:

        mock_igen.return_value.generate.return_value = "input data"

        yield mock_igen, mock_parser, mock_run, mock_write, mock_read

def test_qe_runner_success(mock_runner_deps):
    mock_igen, mock_parser, mock_run, _, mock_read = mock_runner_deps

    # Setup mocks
    mock_parser.return_value.parse.return_value = DFTResult(
        job_id="job1",
        status=JobStatus.COMPLETED,
        work_dir=Path("."),
        duration_seconds=1.0,
        log_content="",
        energy=-10.0,
        forces=np.zeros((1,3))
    )
    mock_run.return_value.stdout = "job done"
    mock_read.return_value = "job done" # Mock reading log file

    runner = QERunner(work_dir=Path("."))
    structure = Structure.from_ase(ase.Atoms("H"))
    config = DFTConfig(command="pw.x", pseudopotentials={}, ecutwfc=10.0)

    result = runner.run(structure, config)

    assert result.status == JobStatus.COMPLETED
    assert result.energy == -10.0
    mock_run.assert_called()

def test_qe_runner_retry(mock_runner_deps):
    mock_igen, mock_parser, mock_run, _, mock_read = mock_runner_deps
    from mlip_autopipec.domain_models.calculation import SCFError

    # First call raises SCFError, second returns result
    mock_parser.return_value.parse.side_effect = [
        SCFError("fail"),
        DFTResult(
            job_id="job1", status=JobStatus.COMPLETED, work_dir=Path("."),
            duration_seconds=1.0, log_content="", energy=-10.0, forces=np.zeros((1,3))
        )
    ]
    mock_read.return_value = "output"

    runner = QERunner(work_dir=Path("."))
    structure = Structure.from_ase(ase.Atoms("H"))
    config = DFTConfig(command="pw.x", pseudopotentials={}, ecutwfc=10.0)

    result = runner.run(structure, config)

    assert result.status == JobStatus.COMPLETED
    assert mock_parser.return_value.parse.call_count == 2
