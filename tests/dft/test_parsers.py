from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from mlip_autopipec.data_models.dft_models import DFTResult
from mlip_autopipec.dft.parsers import QEOutputParser


def test_parse_success():
    mock_reader = MagicMock()
    mock_atoms = MagicMock()
    mock_atoms.get_potential_energy.return_value = -100.0
    mock_atoms.get_forces.return_value = np.zeros((2, 3))
    mock_atoms.get_stress.return_value = np.zeros((3, 3))
    mock_reader.return_value = mock_atoms

    parser = QEOutputParser(reader=mock_reader)
    result = parser.parse(Path("pw.out"), "test-id", 10.0, {})

    assert isinstance(result, DFTResult)
    assert result.energy == -100.0
    assert result.uid == "test-id"


def test_parse_failure():
    mock_reader = MagicMock()
    mock_reader.side_effect = Exception("Read Error")

    parser = QEOutputParser(reader=mock_reader)
    with pytest.raises(Exception) as excinfo:
        parser.parse(Path("pw.out"), "test-id", 10.0, {})
    assert "Failed to parse output" in str(excinfo.value)
