import io
import sys
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

# Mock pyacemaker before importing pace_driver
sys.modules["pyacemaker"] = MagicMock()
sys.modules["pyacemaker.calculator"] = MagicMock()

from mlip_autopipec.inference import pace_driver  # noqa: E402


def test_read_geometry_structure():
    # Input matching EON client format
    # Line 1: number of atoms (N)
    # Line 2: Energy (ignored for input usually)
    # Line 3-5: Box vectors (ax, ay, az; bx, by, bz; cx, cy, cz)
    # Line 6-(6+N): Type X Y Z

    # Wait, standard .con format:
    # Title
    # Box matrix
    # N_atoms
    # Type Mass X Y Z Fixed x_vel y_vel z_vel ...

    # But EON client communication might be simpler.
    # "The client reads the configuration from standard input."
    # EON usually sends:
    # number of atoms
    # box vectors (9 floats)
    # atoms (type x y z)

    # I will assume a format for the test and ensure my implementation matches it.
    # Format:
    # <natoms>
    # <box_x_x> <box_x_y> ... <box_z_z>
    # <type> <x> <y> <z>
    # ...

    input_str = "2\n10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0\nC 0.0 0.0 0.0\nO 1.2 0.0 0.0\n"
    # Verify pace_driver has read_geometry
    if not hasattr(pace_driver, 'read_geometry'):
        pytest.fail("pace_driver.read_geometry not implemented")

    atoms = pace_driver.read_geometry(io.StringIO(input_str))

    assert isinstance(atoms, Atoms)
    assert len(atoms) == 2
    assert atoms.get_chemical_symbols() == ["C", "O"]
    assert atoms.cell[0, 0] == 10.0

def test_write_results():
    # Check output format
    # Expects: Energy
    # Forces (fx fy fz per atom)

    atoms = Atoms("CO", positions=[[0,0,0], [1.2,0,0]])
    atoms.get_potential_energy = MagicMock(return_value=-10.5)
    atoms.get_forces = MagicMock(return_value=[[0,0,1], [0,0,-1]])

    output = io.StringIO()

    if not hasattr(pace_driver, 'print_results'):
        pytest.fail("pace_driver.print_results not implemented")

    pace_driver.print_results(atoms, output)

    result = output.getvalue()
    lines = result.strip().split('\n')
    assert float(lines[0]) == -10.5
    parts = lines[1].split()
    assert abs(float(parts[2]) - 1.0) < 1e-6

@pytest.mark.skipif(sys.version_info < (3, 7), reason="requires python3.7 or higher")
def test_main_execution():
    # Setup mocks
    mock_calc_class = sys.modules["pyacemaker.calculator"].PaceCalculator
    mock_calc_instance = mock_calc_class.return_value
    mock_calc_instance.get_potential_energy.return_value = -10.5
    mock_calc_instance.get_forces.return_value = [[0,0,0], [0,0,0]]

    # But wait, ASE Atoms.get_potential_energy relies on calculator checking 'implemented_properties'
    # or just calling get_potential_energy.
    # Mocking standard ASE calculator behavior on a MagicMock is tricky.
    # Easier: Mock read_geometry to return atoms with a mock calculator already set?
    # No, main calls read_geometry then sets calc.

    # Let's mock atoms.get_potential_energy directly?
    # We can patch 'mlip_autopipec.inference.pace_driver.read_geometry'

    with patch('mlip_autopipec.inference.pace_driver.read_geometry') as mock_read:
        atoms = MagicMock()
        mock_read.return_value = atoms
        atoms.get_potential_energy.return_value = -10.5
        atoms.get_forces.return_value = [[0,0,0]]
        atoms.__len__.return_value = 1

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
             with patch("sys.stdin", new_callable=io.StringIO):
                 pace_driver.main()

        output = mock_stdout.getvalue()
        assert "-1.0500000000000000e+01" in output

def test_read_geometry_empty():
    with pytest.raises(ValueError, match="Empty input"):
        pace_driver.read_geometry(io.StringIO(""))

def test_read_geometry_malformed_box():
    input_str = "2\n1.0 2.0\nAtom..."
    with pytest.raises(ValueError, match="Failed to parse EON geometry"):
        pace_driver.read_geometry(io.StringIO(input_str))

@pytest.mark.skipif(sys.version_info < (3, 7), reason="requires python3.7 or higher")
def test_main_gamma_halt():
    # Setup mocks
    mock_calc_class = sys.modules["pyacemaker.calculator"].PaceCalculator
    mock_calc_instance = mock_calc_class.return_value
    mock_calc_instance.get_potential_energy.return_value = -10.5
    mock_calc_instance.get_forces.return_value = [[0,0,0]] * 2
    mock_calc_instance.results = {"c_gamma_val": 100.0}

    with patch('mlip_autopipec.inference.pace_driver.read_geometry') as mock_read:
        atoms = MagicMock()
        mock_read.return_value = atoms
        atoms.get_potential_energy.return_value = -10.5
        atoms.get_forces.return_value = [[0,0,0]] * 2
        atoms.__len__.return_value = 2

        with patch("sys.stdout", new_callable=io.StringIO):
             with patch("sys.stdin", new_callable=io.StringIO):
                 with patch.dict("os.environ", {"MLIP_GAMMA_THRESHOLD": "1.0"}):
                     with pytest.raises(SystemExit) as exc:
                         pace_driver.main()
                     assert exc.value.code == 100
