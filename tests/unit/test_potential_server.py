import io
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

# We will implement these in src/mlip_autopipec/dynamics/potential_server.py
# For now, we import them to ensure the test fails (Red phase)
try:
    from mlip_autopipec.dynamics.potential_server import (
        format_eon_output,
        parse_eon_input,
        process_structure,
    )
except ImportError:
    pass

@pytest.fixture
def mock_stdin():
    # Example EON input:
    # 0.000000 0.000000 0.000000
    # 10.0 0.0 0.0
    # 0.0 10.0 0.0
    # 0.0 0.0 10.0
    # 2
    # 1.0 2.0 3.0
    # 4.0 5.0 6.0
    # (Note: EON format details might vary, assuming simple cartesian for now or standard EON)
    # Actually, EON usually passes the structure via a file or consistent format.
    # The spec says "Protocol: It reads atomic coordinates from stdin (EON format)"
    # Let's assume a standard XYZ-like or EON specific format.
    # EON usually sends:
    # number_of_atoms
    # ax ay az
    # bx by bz
    # cx cy cz
    # x1 y1 z1
    # ...
    content = """2
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
0.5 0.5 0.5
1.5 1.5 1.5
"""
    return io.StringIO(content)

def test_parse_eon_input(mock_stdin):
    # This should fail initially
    symbols = ["H", "H"]
    atoms = parse_eon_input(mock_stdin.read(), symbols)
    assert isinstance(atoms, Atoms)
    assert len(atoms) == 2
    assert np.allclose(atoms.get_cell(), np.diag([10.0, 10.0, 10.0]))
    assert np.allclose(atoms.get_positions()[0], [0.5, 0.5, 0.5])

def test_format_eon_output():
    energy = -10.5
    forces = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    gamma = 1.2

    output = format_eon_output(energy, forces, gamma)
    # EON expects: Energy line, then Force lines
    lines = output.strip().split('\n')
    assert len(lines) == 3 # 1 energy + 2 forces
    assert str(energy) in lines[0]
    assert "0.1" in lines[1]

def test_process_structure_halt():
    # Test that high uncertainty raises an exception or specific return
    atoms = Atoms('H2', positions=[[0,0,0], [1,0,0]])
    mock_calc = MagicMock()
    mock_calc.get_potential_energy.return_value = -5.0
    mock_calc.get_forces.return_value = np.zeros((2, 3))
    # Mock uncertainty
    mock_calc.results = {'uncertainty': 10.0} # > threshold

    with pytest.raises(SystemExit) as excinfo:
        process_structure(atoms, mock_calc, threshold=5.0)

    assert excinfo.value.code == 100
