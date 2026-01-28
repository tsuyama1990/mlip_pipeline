from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.inference.drivers import pace_driver


# We need to mock sys.exit to prevent test exit
@pytest.fixture
def mock_sys_exit():
    with patch("sys.exit") as mock:
        yield mock

@patch("mlip_autopipec.inference.drivers.pace_driver.read")
@patch("mlip_autopipec.inference.drivers.pace_driver.get_calculator")
@patch("pathlib.Path.exists")
@patch("os.environ.get")
def test_driver_execution_success(mock_env, mock_exists, mock_get_calc, mock_read, mock_sys_exit, capsys):
    # Setup
    mock_exists.return_value = True
    mock_env.side_effect = lambda k: "pot.yace" if k == "PACE_POTENTIAL_PATH" else None

    mock_atoms = MagicMock(spec=Atoms)
    mock_atoms.get_potential_energy.return_value = -10.0
    mock_atoms.get_forces.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_read.return_value = mock_atoms

    mock_calc = MagicMock()
    # We allow get_gamma to exist and return 0.5
    mock_calc.get_gamma.return_value = 0.5

    mock_get_calc.return_value = mock_calc

    # Run
    pace_driver.run_driver()

    # Verify stdout
    captured = capsys.readouterr()
    assert "-10.000000" in captured.out
    assert "0.100000 0.200000 0.300000" in captured.out

    mock_sys_exit.assert_not_called()

@patch("mlip_autopipec.inference.drivers.pace_driver.read")
@patch("mlip_autopipec.inference.drivers.pace_driver.get_calculator")
@patch("pathlib.Path.exists")
@patch("os.environ.get")
def test_driver_execution_halt(mock_env, mock_exists, mock_get_calc, mock_read, mock_sys_exit, capsys):
    # Setup
    mock_exists.return_value = True
    # Return threshold
    mock_env.side_effect = lambda k: "pot.yace" if k == "PACE_POTENTIAL_PATH" else ("5.0" if k == "PACE_GAMMA_THRESHOLD" else None)

    mock_atoms = MagicMock(spec=Atoms)
    mock_atoms.get_potential_energy.return_value = -10.0
    mock_atoms.get_forces.return_value = np.array([[0.0, 0.0, 0.0]])
    mock_read.return_value = mock_atoms

    mock_calc = MagicMock()
    mock_calc.get_gamma.return_value = 10.0 # High gamma
    mock_get_calc.return_value = mock_calc

    # Run
    pace_driver.run_driver()

    captured = capsys.readouterr()
    assert "Halt: Gamma 10.0" in captured.err

    mock_sys_exit.assert_called_with(100)
