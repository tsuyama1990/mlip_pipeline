from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mlip_autopipec.components.dynamics.lammps import LAMMPSDynamics
from mlip_autopipec.components.dynamics.lammps_driver import LAMMPSDriver
from mlip_autopipec.domain_models.config import LAMMPSDynamicsConfig
from mlip_autopipec.domain_models.enums import DynamicsType
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure


@pytest.fixture
def structure() -> Structure:
    return Structure(
        positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        atomic_numbers=np.array([29, 29]),
        cell=np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]),
        pbc=np.array([True, True, True]),
    )


@pytest.fixture
def potential(tmp_path: Path) -> Potential:
    return Potential(path=tmp_path / "test.yace", species=["Cu"], format="yace")


@pytest.fixture
def config() -> LAMMPSDynamicsConfig:
    return LAMMPSDynamicsConfig(
        name=DynamicsType.LAMMPS,
        n_steps=1000,
        timestep=0.002,
        uncertainty_threshold=1.5
    )

def test_lammps_driver_write_input(
    tmp_path: Path, structure: Structure, potential: Potential, config: LAMMPSDynamicsConfig
) -> None:
    driver = LAMMPSDriver(workdir=tmp_path)
    driver.write_input_files(structure, potential, config)

    # Check data.lammps
    data_file = tmp_path / "data.lammps"
    assert data_file.exists()
    content = data_file.read_text()
    assert "2 atoms" in content
    assert "1 atom types" in content
    assert "xlo xhi" in content
    assert "0.0" in content

    # Check in.lammps
    input_file = tmp_path / "in.lammps"
    assert input_file.exists()
    content = input_file.read_text()
    assert "pair_style pace" in content
    assert "fix halt" in content
    assert "halt" in content
    assert "v_max_gamma > 1.5" in content  # Check threshold
    # Note: run 1000 might be on a separate line
    assert "run" in content
    assert "1000" in content


def test_lammps_driver_parse_log_halted(tmp_path: Path) -> None:
    log_content = """
Step Temp PotEng max_gamma
0 300 -6.0 0.1
10 300 -6.0 1.6
ERROR: Halt: max_gamma > 1.5
"""
    (tmp_path / "log.lammps").write_text(log_content)

    driver = LAMMPSDriver(workdir=tmp_path)
    result = driver.parse_log()
    assert result["halted"] is True
    assert result["final_step"] == 10


def test_lammps_driver_parse_log_finished(tmp_path: Path) -> None:
    log_content = """
Step Temp PotEng max_gamma
0 300 -6.0 0.1
1000 300 -6.0 0.2
Loop time of 1.23 on 1 procs
"""
    (tmp_path / "log.lammps").write_text(log_content)

    driver = LAMMPSDriver(workdir=tmp_path)
    result = driver.parse_log()
    assert result["halted"] is False
    assert result["final_step"] == 1000


@patch("mlip_autopipec.components.dynamics.lammps.LAMMPSDriver")
def test_lammps_dynamics_explore(
    mock_driver_cls: MagicMock,
    tmp_path: Path,
    structure: Structure,
    potential: Potential,
    config: LAMMPSDynamicsConfig
) -> None:
    dynamics = LAMMPSDynamics(config)

    # Mock driver instance
    mock_driver = mock_driver_cls.return_value
    mock_driver.write_input_files = MagicMock()
    mock_driver.run_md = MagicMock()
    mock_driver.parse_log.return_value = {"halted": True, "final_step": 50}

    # Mock read_dump returning a structure
    # We need a structure with correct number of atoms
    dump_struct = structure.model_deep_copy()
    dump_struct.positions += 0.1 # Moved
    mock_driver.read_dump.return_value = dump_struct

    # Run explore
    results = list(dynamics.explore(potential, [structure], workdir=tmp_path))

    assert len(results) == 1
    assert results[0].uncertainty == 100.0
    assert results[0].tags["provenance"] == "dynamics_halted"

    mock_driver.write_input_files.assert_called_once()
    mock_driver.run_md.assert_called_once()
    mock_driver.parse_log.assert_called_once()
    mock_driver.read_dump.assert_called_once()
