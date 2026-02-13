from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from mlip_autopipec.domain_models.config import DynamicsConfig, EONConfig
from mlip_autopipec.domain_models.datastructures import Potential, Structure
from mlip_autopipec.domain_models.enums import DynamicsType

# We will implement this in src/mlip_autopipec/dynamics/eon_driver.py
try:
    from mlip_autopipec.dynamics.eon_driver import EONDriver
except ImportError:
    pass

@pytest.fixture
def mock_config():
    eon = EONConfig(
        temperature=300.0,
        prefactor=1e12,
        search_method="akmc",
        client_path="eonclient",
        server_script_name="potential_server.py"
    )
    return DynamicsConfig(type=DynamicsType.EON, eon=eon)

@pytest.fixture
def mock_potential():
    return Potential(
        path=Path("/tmp/potential.yace"),
        format="yace"
    )

@pytest.fixture
def mock_structure():
    from ase.build import bulk
    atoms = bulk("Cu", "fcc", a=3.6)
    return Structure(
        atoms=atoms,
        provenance="test",
        label_status="unlabeled"
    )

def test_eon_driver_init(mock_config):
    work_dir = Path("/tmp/work_dir")
    driver = EONDriver(work_dir, mock_config)
    assert driver.work_dir == work_dir
    assert driver.config == mock_config

@patch("subprocess.run")
@patch("builtins.open", new_callable=mock_open)
@patch("shutil.rmtree")
@patch("pathlib.Path.mkdir")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.write_text")
@patch("mlip_autopipec.dynamics.eon_driver.write")
def test_eon_driver_simulate(mock_ase_write, mock_write_text, mock_exists, mock_mkdir, mock_rmtree, mock_file, mock_subprocess, mock_config, mock_potential, mock_structure):
    work_dir = Path("/tmp/work_dir")
    driver = EONDriver(work_dir, mock_config)

    # Mock subprocess success
    mock_subprocess.return_value.returncode = 0

    # Mock result reading
    mock_exists.return_value = True

    with patch("shutil.copy"):
        structures = list(driver.simulate(mock_potential, mock_structure))

    mock_subprocess.assert_called_once()
    args = mock_subprocess.call_args[0][0]
    assert "eonclient" in args

@patch("pathlib.Path.write_text")
@patch("mlip_autopipec.dynamics.eon_driver.write")
def test_eon_driver_prepare_run(mock_ase_write, mock_write_text, mock_config, mock_potential, mock_structure):
    work_dir = Path("/tmp/work_dir")
    driver = EONDriver(work_dir, mock_config)

    with patch("shutil.copy"):
        driver._prepare_run(mock_potential, mock_structure, work_dir)

    # Check if config.ini was written
    mock_write_text.assert_called()

@patch("shutil.rmtree")
@patch("pathlib.Path.mkdir")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.write_text")
@patch("mlip_autopipec.dynamics.eon_driver.write")
@patch("pathlib.Path.read_text")
@patch("mlip_autopipec.dynamics.eon_driver.read")
def test_eon_driver_halt_detection(mock_read, mock_read_text, mock_ase_write, mock_write_text, mock_exists, mock_mkdir, mock_rmtree, mock_config, mock_potential, mock_structure):
    work_dir = Path("/tmp/work_dir")
    driver = EONDriver(work_dir, mock_config)

    # Mock subprocess returning 100 (halt code)
    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value.returncode = 100

        mock_exists.return_value = True
        mock_read_text.return_value = "max_gamma: 10.0"
        mock_read.return_value = mock_structure.atoms

        with patch("shutil.copy"):
             results = list(driver.simulate(mock_potential, mock_structure))
             assert len(results) > 0
             assert results[-1].metadata.get("halt_reason") == "uncertainty"
