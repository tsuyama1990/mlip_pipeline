from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.dft.parsers import QEOutputParser


@pytest.fixture
def mock_read():
    with patch("mlip_autopipec.dft.parsers.ase_read") as m:
        yield m

def test_parser_success(tmp_path, mock_read):
    output_file = tmp_path / "pw.out"
    output_file.write_text("... JOB DONE ...")

    mock_atoms = MagicMock(spec=Atoms)
    mock_atoms.get_potential_energy.return_value = -10.0
    mock_atoms.get_forces.return_value = np.zeros((2,3))
    mock_atoms.get_stress.return_value = np.zeros((3,3))
    mock_read.return_value = mock_atoms

    parser = QEOutputParser(reader=mock_read)
    result = parser.parse(output_file, "uid", 1.0, {})

    assert result.energy == -10.0
    assert result.succeeded

def test_parser_missing_job_done(tmp_path, mock_read):
    output_file = tmp_path / "pw.out"
    output_file.write_text("... CRASH ...")

    parser = QEOutputParser(reader=mock_read)
    with pytest.raises(Exception, match="JOB DONE"):
        parser.parse(output_file, "uid", 1.0, {})

def test_parser_file_not_found(tmp_path, mock_read):
    output_file = tmp_path / "nonexistent.out"
    parser = QEOutputParser(reader=mock_read)
    with pytest.raises(Exception, match="not found"):
        parser.parse(output_file, "uid", 1.0, {})

def test_parser_nan_forces(tmp_path, mock_read):
    output_file = tmp_path / "pw.out"
    output_file.write_text("JOB DONE")

    mock_atoms = MagicMock(spec=Atoms)
    mock_atoms.get_potential_energy.return_value = -10.0
    mock_atoms.get_forces.return_value = np.array([[np.nan, 0, 0]])
    mock_atoms.get_stress.return_value = np.zeros((3,3))
    mock_read.return_value = mock_atoms

    parser = QEOutputParser(reader=mock_read)
    # The parser wraps exceptions
    with pytest.raises(Exception, match="NaN or Inf"):
        parser.parse(output_file, "uid", 1.0, {})
