import contextlib
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

with contextlib.suppress(ImportError):
    from mlip_autopipec.dynamics.potential_server import (
        format_eon_output,
        parse_eon_input,
        process_structure,
    )

@pytest.fixture
def mock_stdin() -> io.StringIO:
    content = """2
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
0.5 0.5 0.5
1.5 1.5 1.5
"""
    return io.StringIO(content)

def test_parse_eon_input(mock_stdin: io.StringIO) -> None:
    symbols = ["H", "H"]
    atoms = parse_eon_input(mock_stdin.read(), symbols)
    assert isinstance(atoms, Atoms)
    assert len(atoms) == 2
    assert np.allclose(atoms.get_cell(), np.diag([10.0, 10.0, 10.0]))

def test_parse_eon_input_empty() -> None:
    with pytest.raises(ValueError, match="Empty input"):
        parse_eon_input("", ["H"])

def test_parse_eon_input_malformed_header() -> None:
    content = "nan\n"
    with pytest.raises(ValueError, match="Invalid EON input"):
        parse_eon_input(content, ["H"])

def test_parse_eon_input_negative_atoms() -> None:
    content = "-1\n"
    with pytest.raises(ValueError, match="Negative atom count"):
        parse_eon_input(content, [])

def test_format_eon_output() -> None:
    energy = -10.5
    forces = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    gamma = 1.2

    output = format_eon_output(energy, forces, gamma)
    lines = output.strip().split('\n')
    assert len(lines) == 3
    assert str(energy) in lines[0]
    assert "0.1" in lines[1]

def test_process_structure_halt() -> None:
    atoms = Atoms('H2', positions=[[0,0,0], [1,0,0]])
    mock_calc = MagicMock()
    mock_calc.get_potential_energy.return_value = -5.0
    mock_calc.get_forces.return_value = np.zeros((2, 3))
    mock_calc.results = {'uncertainty': 10.0}

    # Mock sys.exit and file writing
    with patch("sys.exit") as mock_exit, \
         patch("mlip_autopipec.dynamics.potential_server.write"), \
         patch("pathlib.Path.open", new_callable=MagicMock):
             process_structure(atoms, mock_calc, threshold=5.0)
             mock_exit.assert_called_with(100)

def test_process_structure_success() -> None:
    atoms = Atoms('H2', positions=[[0,0,0], [1,0,0]])
    mock_calc = MagicMock()
    mock_calc.get_potential_energy.return_value = -5.0
    mock_calc.get_forces.return_value = np.zeros((2, 3))
    mock_calc.results = {'uncertainty': 1.0}

    e, f, g = process_structure(atoms, mock_calc, threshold=5.0)
    assert e == -5.0
    assert g == 1.0
