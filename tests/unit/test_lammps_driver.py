from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.domain_models.config import DynamicsConfig
from mlip_autopipec.domain_models.datastructures import Potential, Structure
from mlip_autopipec.domain_models.enums import DynamicsType
from mlip_autopipec.dynamics.lammps_driver import LAMMPSDriver


@pytest.fixture
def dynamics_config() -> DynamicsConfig:
    return DynamicsConfig(
        type=DynamicsType.LAMMPS,
        temperature=1000.0,
        steps=500,
        timestep=0.001,
        halt_on_uncertainty=True,
        max_gamma_threshold=5.0,
        n_dump=10,
        n_thermo=10,
    )


@pytest.fixture
def mock_structure() -> Structure:
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2, 0, 0]], cell=[10, 10, 10], pbc=True)
    return Structure(atoms=atoms, provenance="test", label_status="unlabeled")


@pytest.fixture
def mock_potential(tmp_path: Path) -> Potential:
    p = tmp_path / "potential.yace"
    p.touch()
    return Potential(path=p, format="yace")


@patch("subprocess.run")
@patch("mlip_autopipec.dynamics.lammps_driver.write")
@patch("mlip_autopipec.dynamics.lammps_driver.iread")
def test_simulate_success(
    mock_iread: MagicMock,
    mock_write: MagicMock,
    mock_run: MagicMock,
    dynamics_config: DynamicsConfig,
    mock_structure: Structure,
    mock_potential: Potential,
    tmp_path: Path,
) -> None:
    driver = LAMMPSDriver(tmp_path, dynamics_config)

    def create_dump_file(*args: Any, **kwargs: Any) -> MagicMock:
        # Create the dump file to simulate LAMMPS output
        # Driver creates "md_run_test" dir because provenance is "test"
        dump = tmp_path / "md_run_test" / "traj.dump"
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.touch()
        return MagicMock(returncode=0, stdout="Simulation done")

    mock_run.side_effect = create_dump_file

    # Mock trajectory reading
    frame1 = mock_structure.atoms.copy() # type: ignore[no-untyped-call]
    frame1.new_array('c_pace[1]', np.array([0.1, 0.2])) # Low gamma
    # Simulate ASE reading 'type' column as atomic numbers (so 1-based index)
    frame1.set_atomic_numbers([1, 1]) # type: ignore[no-untyped-call]

    frame2 = mock_structure.atoms.copy() # type: ignore[no-untyped-call]
    frame2.positions[0] += 0.1
    frame2.new_array('c_pace[1]', np.array([0.2, 0.3])) # Low gamma
    frame2.set_atomic_numbers([1, 1]) # type: ignore[no-untyped-call]

    mock_iread.return_value = iter([frame1, frame2])

    trajectory = list(driver.simulate(mock_potential, mock_structure))

    assert len(trajectory) == 2
    assert trajectory[0].provenance == "md_trajectory"
    assert trajectory[0].uncertainty_score == 0.2
    assert trajectory[1].uncertainty_score == 0.3

    mock_write.assert_called()
    mock_run.assert_called()


@patch("subprocess.run")
@patch("mlip_autopipec.dynamics.lammps_driver.write")
@patch("mlip_autopipec.dynamics.lammps_driver.iread")
def test_simulate_halt_event(
    mock_iread: MagicMock,
    mock_write: MagicMock,
    mock_run: MagicMock,
    dynamics_config: DynamicsConfig,
    mock_structure: Structure,
    mock_potential: Potential,
    tmp_path: Path,
) -> None:
    driver = LAMMPSDriver(tmp_path, dynamics_config)

    def create_dump_file_halt(*args: Any, **kwargs: Any) -> MagicMock:
        dump = tmp_path / "md_run_test" / "traj.dump"
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.touch()
        # Raise CalledProcessError to simulate check=True failure
        import subprocess
        raise subprocess.CalledProcessError(100, args[0])

    mock_run.side_effect = create_dump_file_halt

    # Mock trajectory reading - halted frame has high gamma
    frame1 = mock_structure.atoms.copy() # type: ignore[no-untyped-call]
    frame1.new_array('c_pace[1]', np.array([1.0, 6.0])) # High gamma > 5.0
    frame1.set_atomic_numbers([1, 1]) # type: ignore[no-untyped-call]

    mock_iread.return_value = iter([frame1])

    trajectory = list(driver.simulate(mock_potential, mock_structure))

    assert len(trajectory) == 1
    assert trajectory[0].uncertainty_score == 6.0

    mock_run.assert_called()
