import pytest
from unittest.mock import MagicMock
from pathlib import Path
import numpy as np
from ase import Atoms

from mlip_autopipec.dft.parsers import QEOutputParser
from mlip_autopipec.data_models.dft_models import DFTResult

def test_parser_success(tmp_path):
    mock_atoms = MagicMock(spec=Atoms)
    mock_atoms.get_potential_energy.return_value = -123.456
    mock_atoms.get_forces.return_value = np.array([[0.0, 0.0, 0.1]])
    mock_atoms.get_stress.return_value = np.zeros((3, 3))

    mock_reader = MagicMock(return_value=mock_atoms)

    parser = QEOutputParser(reader=mock_reader)

    out_file = tmp_path / "pw.out"
    out_file.write_text("JOB DONE")

    result = parser.parse(
        out_file,
        uid="test-1",
        wall_time=10.0,
        params={"ecutwfc": 30}
    )

    assert result.succeeded
    assert result.energy == -123.456

def test_parser_incomplete_file(tmp_path):
    out_file = tmp_path / "pw.out"
    out_file.write_text("Still running...")

    parser = QEOutputParser()

    with pytest.raises(Exception, match="missing 'JOB DONE'"):
        parser.parse(out_file, "uid", 1.0, {})

def test_parser_ase_failure(tmp_path):
    out_file = tmp_path / "pw.out"
    out_file.write_text("JOB DONE")

    mock_reader = MagicMock(side_effect=Exception("ASE Parse Error"))
    parser = QEOutputParser(reader=mock_reader)

    with pytest.raises(Exception, match="Failed to parse output"):
        parser.parse(out_file, "uid", 1.0, {})

def test_parser_nan_forces(tmp_path):
    mock_atoms = MagicMock(spec=Atoms)
    mock_atoms.get_potential_energy.return_value = -100.0
    mock_atoms.get_forces.return_value = np.array([[np.nan, 0.0, 0.0]])
    mock_atoms.get_stress.return_value = np.zeros((3, 3))

    mock_reader = MagicMock(return_value=mock_atoms)
    parser = QEOutputParser(reader=mock_reader)

    out_file = tmp_path / "pw.out"
    out_file.write_text("JOB DONE")

    with pytest.raises(Exception, match="Failed to parse output"):
        parser.parse(out_file, "uid", 1.0, {})

def test_parser_file_not_found():
    parser = QEOutputParser()
    with pytest.raises(Exception, match="Output file not found"):
        parser.parse(Path("nonexistent"), "uid", 1.0, {})
