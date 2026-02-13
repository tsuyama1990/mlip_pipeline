import contextlib
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

# Import directly or use placeholders if import fails
with contextlib.suppress(ImportError):
    from mlip_autopipec.dynamics.potential_server import (
        format_eon_output,
        parse_eon_input,
        process_structure,
    )

@pytest.fixture
def valid_eon_input_stream() -> Iterator[str]:
    content = """2
0.0
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
0.5 0.5 0.5
1.5 1.5 1.5
"""
    return iter(content.splitlines())

def test_parse_eon_input(valid_eon_input_stream: Iterator[str]) -> None:
    symbols = ["H", "H"]
    atoms = parse_eon_input(valid_eon_input_stream, symbols)
    assert isinstance(atoms, Atoms)
    assert len(atoms) == 2
    # Check cell (diagonal)
    cell = atoms.get_cell() # type: ignore[no-untyped-call]
    assert np.allclose(cell, np.diag([10.0, 10.0, 10.0]))

def test_parse_eon_input_empty() -> None:
    empty_stream: Iterator[str] = iter([])
    with pytest.raises(ValueError, match="Empty input"):
        parse_eon_input(empty_stream, ["H"])

def test_parse_eon_input_malformed_header() -> None:
    stream = iter(["nan"])
    with pytest.raises(ValueError, match="Invalid EON input"):
        parse_eon_input(stream, ["H"])

def test_parse_eon_input_negative_atoms() -> None:
    stream = iter(["-1"])
    with pytest.raises(ValueError, match="Negative atom count"):
        parse_eon_input(stream, [])

def test_format_eon_output() -> None:
    energy = -10.5
    forces = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    gamma = 1.2

    # Updated to match actual implementation logic if changed
    output = format_eon_output(energy, forces, gamma)
    lines = output.strip().split('\n')
    # 1 line for energy + 2 lines for forces
    assert len(lines) == 3
    assert f"{energy:.6f}" in lines[0]
    assert "0.100000" in lines[1]

def test_process_structure_halt() -> None:
    atoms = Atoms('H2', positions=[[0,0,0], [1,0,0]])
    mock_calc = MagicMock()
    mock_calc.get_potential_energy.return_value = -5.0
    mock_calc.get_forces.return_value = np.zeros((2, 3))
    mock_calc.results = {'uncertainty': 10.0}
    atoms.calc = mock_calc

    # Mock sys.exit and file writing
    # Note: process_structure uses sys.exit(100) on high uncertainty
    with (
        patch("sys.exit") as mock_exit,
        patch("mlip_autopipec.dynamics.potential_server.write") as mock_write,
        patch("pathlib.Path.open", new_callable=MagicMock),
    ):
         process_structure(atoms, mock_calc, threshold=5.0)
         mock_exit.assert_called_with(100)
         mock_write.assert_called()

def test_process_structure_success() -> None:
    atoms = Atoms('H2', positions=[[0,0,0], [1,0,0]])
    mock_calc = MagicMock()
    mock_calc.get_potential_energy.return_value = -5.0
    mock_calc.get_forces.return_value = np.zeros((2, 3))
    mock_calc.results = {'uncertainty': 1.0}
    atoms.calc = mock_calc

    # Should return (energy, forces, gamma)
    e, f, g = process_structure(atoms, mock_calc, threshold=5.0)
    assert e == -5.0
    assert g == 1.0
