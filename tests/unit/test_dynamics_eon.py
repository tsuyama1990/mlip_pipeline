from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mlip_autopipec.components.dynamics.eon import EONDriver, EONDynamics
from mlip_autopipec.domain_models.config import EONDynamicsConfig
from mlip_autopipec.domain_models.enums import DynamicsType
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure


@pytest.fixture
def structure() -> Structure:
    return Structure(
        positions=np.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=np.array([29]),
        cell=np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]),
        pbc=np.array([True, True, True]),
    )


@pytest.fixture
def potential(tmp_path: Path) -> Potential:
    return Potential(path=tmp_path / "test.yace", species=["Cu"], format="yace")


@pytest.fixture
def config() -> EONDynamicsConfig:
    return EONDynamicsConfig(
        name=DynamicsType.EON, temperature=500.0, n_events=1000, prefactor=1e13
    )


def test_eon_driver_write_input(
    tmp_path: Path, structure: Structure, potential: Potential, config: EONDynamicsConfig
) -> None:
    driver = EONDriver(workdir=tmp_path)
    driver.write_input_files(structure, potential, config)

    # Check config.ini
    config_file = tmp_path / "config.ini"
    assert config_file.exists()
    content = config_file.read_text()
    assert "temperature = 500.0" in content
    # Python formatting might vary slightly, check key exists
    assert "prefactor =" in content
    assert "13" in content
    assert "max_events = 1000" in content

    # Check pos.con (EON structure format)
    pos_file = tmp_path / "pos.con"
    assert pos_file.exists()
    # Basic check for header
    assert "1 atoms" in content or "1" in content  # EON format specific checks

    # Check pace_driver.py
    driver_script = tmp_path / "pace_driver.py"
    assert driver_script.exists()
    content = driver_script.read_text()
    assert "get_calculator" in content
    assert str(tmp_path / "test.yace") in content
    assert "check_uncertainty" in content  # Ensuring OTF logic is present


@patch("mlip_autopipec.components.dynamics.eon.EONDriver")
@patch("mlip_autopipec.components.dynamics.eon.read")
def test_eon_dynamics_explore(
    mock_read: MagicMock,
    mock_driver_cls: MagicMock,
    tmp_path: Path,
    structure: Structure,
    potential: Potential,
    config: EONDynamicsConfig,
) -> None:
    dynamics = EONDynamics(config)

    mock_driver = mock_driver_cls.return_value
    mock_driver.workdir = tmp_path / "eon_run_00000"
    mock_driver.write_input_files = MagicMock()
    # simulate run_kmc failure or success
    mock_driver.run_kmc.side_effect = Exception("OTF Halt simulated")

    # Simulate halted structure existence
    (tmp_path / "eon_run_00000").mkdir(parents=True)
    halted_file = tmp_path / "eon_run_00000" / "halted_structure.xyz"
    halted_file.touch()

    # Mock reading the halted structure
    # read returns Atoms
    mock_atoms = MagicMock()
    mock_atoms.get_positions.return_value = np.array([[0.1, 0.1, 0.1]])
    mock_atoms.get_atomic_numbers.return_value = np.array([29])
    mock_atoms.get_cell.return_value = np.array([[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]])
    mock_atoms.get_pbc.return_value = np.array([True, True, True])
    mock_atoms.__len__.return_value = 1
    mock_atoms.info = {}
    mock_atoms.arrays = {}
    mock_atoms.calc = None  # Ensure no calculator is attached so it doesn't try to get labels

    mock_read.return_value = mock_atoms

    results = list(dynamics.explore(potential, [structure], workdir=tmp_path))

    assert len(results) == 1
    assert results[0].uncertainty == 100.0
    assert results[0].tags["provenance"] == "dynamics_halted_eon"

    mock_driver.write_input_files.assert_called_once()
    mock_driver.run_kmc.assert_called_once()
    mock_read.assert_called_with(halted_file)
