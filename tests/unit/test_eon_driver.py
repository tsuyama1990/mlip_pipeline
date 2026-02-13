from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from ase.build import bulk

from mlip_autopipec.domain_models.config import DynamicsConfig, EONConfig
from mlip_autopipec.domain_models.datastructures import Potential, Structure
from mlip_autopipec.domain_models.enums import DynamicsType
from mlip_autopipec.dynamics.eon_driver import EONDriver, EONExecutionError


@pytest.fixture
def mock_config() -> DynamicsConfig:
    eon = EONConfig(
        temperature=300.0,
        prefactor=1e12,
        search_method="akmc",
        client_path="eonclient",
        server_script_name="potential_server.py"
    )
    return DynamicsConfig(type=DynamicsType.EON, eon=eon)

@pytest.fixture
def mock_potential(tmp_path: Path) -> Potential:
    p = tmp_path / "potential.yace"
    p.touch()
    return Potential(
        path=p,
        format="yace"
    )

@pytest.fixture
def mock_structure() -> Structure:
    atoms = bulk("Cu", "fcc", a=3.6)
    return Structure(
        atoms=atoms,
        provenance="test",
        label_status="unlabeled"
    )

def test_eon_driver_init(mock_config: DynamicsConfig, tmp_path: Path) -> None:
    work_dir = tmp_path / "work_dir"
    driver = EONDriver(work_dir, mock_config)
    assert driver.work_dir == work_dir
    assert driver.config == mock_config

def test_eon_driver_init_fail_no_eon_config(tmp_path: Path) -> None:
    # Need to override validator or just pass valid config structure with None eon if allowed?
    # DynamicsConfig validator ensures eon is present if type is EON usually.
    # But if not, EONDriver checks it.
    conf = DynamicsConfig(type=DynamicsType.EON, eon=None)
    with pytest.raises(ValueError, match="EON configuration missing"):
        EONDriver(tmp_path, conf)

@patch("subprocess.run")
@patch("builtins.open", new_callable=mock_open)
@patch("shutil.rmtree")
@patch("pathlib.Path.mkdir")
@patch("pathlib.Path.write_text")
@patch("mlip_autopipec.dynamics.eon_driver.write")
def test_eon_driver_simulate(mock_ase_write: MagicMock, mock_write_text: MagicMock, mock_mkdir: MagicMock, mock_rmtree: MagicMock, mock_file: MagicMock, mock_subprocess: MagicMock, mock_config: DynamicsConfig, mock_potential: Potential, mock_structure: Structure, tmp_path: Path) -> None:
    work_dir = tmp_path / "work_dir"
    driver = EONDriver(work_dir, mock_config)

    # Mock subprocess success
    mock_subprocess.return_value.returncode = 0

    # We need to ensure client_path check passes.
    # If client_path is relative ("eonclient"), checks might be skipped or simple.
    # But _prepare_run calls validate_path_safety which checks existence of potential path.
    # Potential path exists due to fixture.

    # We patch shutil.copy
    with patch("shutil.copy"):
        # We need to mock glob to return something? Or empty is fine.
        with patch("pathlib.Path.glob", return_value=[]):
             # We also need to patch Path.exists inside simulate?
             # simulate calls _prepare_run -> potential.path.exists() (Real file exists)
             # simulate calls client_path.exists() if absolute. "eonclient" is relative.
             # simulate calls run_dir.exists(). run_dir is constructed from work_dir.
             # work_dir is real tmp_path. run_dir might not exist.
             _ = list(driver.simulate(mock_potential, mock_structure))

    mock_subprocess.assert_called_once()
    args = mock_subprocess.call_args[0][0]
    assert "eonclient" in args

@patch("pathlib.Path.write_text")
@patch("mlip_autopipec.dynamics.eon_driver.write")
def test_eon_driver_prepare_run(mock_ase_write: MagicMock, mock_write_text: MagicMock, mock_config: DynamicsConfig, mock_potential: Potential, mock_structure: Structure, tmp_path: Path) -> None:
    work_dir = tmp_path / "work_dir"
    driver = EONDriver(work_dir, mock_config)

    # Ensure work_dir exists or is handled
    # _prepare_run uses run_dir = work_dir (passed as arg in test? No, simulate creates run_dir)
    # The test passes work_dir directly to _prepare_run as 'run_dir'

    with patch("shutil.copy"):
        driver._prepare_run(mock_potential, mock_structure, work_dir)

    # Check if config.ini was written
    mock_write_text.assert_called()

@patch("subprocess.run")
@patch("shutil.rmtree")
@patch("pathlib.Path.mkdir")
@patch("pathlib.Path.write_text")
@patch("mlip_autopipec.dynamics.eon_driver.write")
@patch("pathlib.Path.read_text")
@patch("mlip_autopipec.dynamics.eon_driver.read")
def test_eon_driver_halt_detection(mock_read: MagicMock, mock_read_text: MagicMock, mock_ase_write: MagicMock, mock_write_text: MagicMock, mock_mkdir: MagicMock, mock_rmtree: MagicMock, mock_subprocess: MagicMock, mock_config: DynamicsConfig, mock_potential: Potential, mock_structure: Structure, tmp_path: Path) -> None:
    work_dir = tmp_path / "work_dir"
    driver = EONDriver(work_dir, mock_config)

    # Mock subprocess returning 100 (halt code)
    mock_subprocess.return_value.returncode = 100

    # Mock reading halt info
    mock_read_text.return_value = "reason: uncertainty\nmax_gamma: 10.0"

    # Mock reading bad_structure.xyz
    mock_read.return_value = mock_structure.atoms # Return Atoms object

    # We need to ensure bad_structure.xyz exists check passes?
    # _parse_halt checks if bad_struct_path.exists().
    # We need to patch Path.exists to return True for bad_structure.xyz

    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = True # All exist

        with patch("shutil.copy"):
             results = list(driver.simulate(mock_potential, mock_structure))
             assert len(results) > 0
             assert results[-1].metadata.get("halt_reason") == "uncertainty"

@patch("subprocess.run")
@patch("shutil.rmtree")
@patch("pathlib.Path.mkdir")
@patch("pathlib.Path.write_text")
@patch("mlip_autopipec.dynamics.eon_driver.write")
def test_eon_driver_failure(mock_ase_write: MagicMock, mock_write_text: MagicMock, mock_mkdir: MagicMock, mock_rmtree: MagicMock, mock_subprocess: MagicMock, mock_config: DynamicsConfig, mock_potential: Potential, mock_structure: Structure, tmp_path: Path) -> None:
    work_dir = tmp_path / "work_dir"
    driver = EONDriver(work_dir, mock_config)

    # Mock failure
    mock_subprocess.return_value.returncode = 1
    mock_subprocess.return_value.stderr = "Fatal Error"

    with patch("shutil.copy"), pytest.raises(EONExecutionError, match="EON execution failed"):
        # We need Path.exists for potential check in _prepare_run?
        # Potential file exists on disk (fixture), but we need to ensure checks pass
        list(driver.simulate(mock_potential, mock_structure))
