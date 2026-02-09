from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mlip_autopipec.components.dynamics.eon import EONDriver, EONDynamics
from mlip_autopipec.components.dynamics.eon import _run_single_eon_simulation
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
    p = tmp_path / "test.yace"
    p.touch()
    return Potential(path=p, species=["Cu"], format="yace")


@pytest.fixture
def config() -> EONDynamicsConfig:
    return EONDynamicsConfig(
        name=DynamicsType.EON, temperature=500.0, n_events=1000, prefactor=1e13
    )


def test_eon_driver_write_input(
    tmp_path: Path, structure: Structure, potential: Potential, config: EONDynamicsConfig
) -> None:
    driver = EONDriver(workdir=tmp_path, config=config)
    driver.write_input_files(structure, potential)

    # Check config.ini
    config_file = tmp_path / config.config_filename
    assert config_file.exists()
    content = config_file.read_text()
    assert "temperature = 500.0" in content
    # Python formatting might vary slightly, check key exists
    assert "prefactor =" in content
    assert "13" in content
    assert "max_events = 1000" in content

    # Check pos.con (EON structure format)
    pos_file = tmp_path / config.pos_filename
    assert pos_file.exists()

    # Check pace_driver.py
    driver_script = tmp_path / config.driver_filename
    assert driver_script.exists()
    content = driver_script.read_text()
    assert "get_calculator" in content
    assert str(potential.path.resolve()) in content
    assert "check_uncertainty" in content  # Ensuring OTF logic is present

    # Check OTF logic details (check_uncertainty implementation)
    assert "gamma = atoms.info.get('uncertainty', 0.0)" in content
    assert "if gamma > threshold:" in content
    assert "return True" in content


@patch("mlip_autopipec.components.dynamics.eon.subprocess.run")
@patch("mlip_autopipec.components.dynamics.eon.read")
def test_eon_dynamics_explore(
    mock_read: MagicMock,
    mock_subprocess_run: MagicMock,
    tmp_path: Path,
    structure: Structure,
    potential: Potential,
    config: EONDynamicsConfig,
) -> None:
    # Use context manager for Executor mocking to allow sync execution or verifying submission
    with patch("concurrent.futures.ProcessPoolExecutor") as mock_executor_cls:
        mock_executor = mock_executor_cls.return_value
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None

        mock_future = MagicMock()

        # Prepare result structure for the future result
        mock_struct = structure.model_deep_copy()
        mock_struct.uncertainty = 100.0
        mock_struct.tags["provenance"] = "dynamics_halted_eon"

        mock_future.result.return_value = mock_struct

        # as_completed yields futures
        def side_effect_as_completed(fs: list[Any]) -> list[Any]:
            return fs

        with patch("concurrent.futures.as_completed", side_effect=side_effect_as_completed):
            mock_executor.submit.return_value = mock_future

            dynamics = EONDynamics(config)

            results = list(dynamics.explore(potential, [structure], workdir=tmp_path))

            assert len(results) == 1
            assert results[0].uncertainty == 100.0

            # Verify EONDriver methods were not called directly in main process (concurrency check)
            assert mock_executor.submit.call_count == 1


@patch("mlip_autopipec.components.dynamics.eon.subprocess.run")
@patch("mlip_autopipec.components.dynamics.eon.read")
def test_run_single_eon_simulation(
    mock_read: MagicMock,
    mock_subprocess_run: MagicMock,
    tmp_path: Path,
    structure: Structure,
    potential: Potential,
    config: EONDynamicsConfig,
) -> None:
    # Setup mocks for file reading
    # We must ensure mock_atoms behaves like an ASE Atoms object that Structure.from_ase can handle
    mock_atoms = MagicMock()
    mock_atoms.get_positions.return_value = np.array([[0.1, 0.1, 0.1]])
    mock_atoms.get_atomic_numbers.return_value = np.array([29])
    mock_atoms.get_cell.return_value = np.array([[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]])
    mock_atoms.get_pbc.return_value = np.array([True, True, True])
    mock_atoms.__len__.return_value = 1

    # Crucial: Mock info and arrays to be dicts, and calc/labels to avoid validation errors
    mock_atoms.info = {"uncertainty": 100.0}
    mock_atoms.arrays = {}
    mock_atoms.calc = None # No calculator means get_potential_energy etc might raise or return None if checked differently
    # But Structure.from_ase calls get_potential_energy if calc exists.
    # If calc is None, it checks info/arrays.

    mock_read.return_value = mock_atoms

    # Mock subprocess to simulate EON run
    mock_subprocess_run.return_value = MagicMock(returncode=0)

    # We need to simulate the existence of the halted file
    idx = 0
    run_dir = tmp_path / f"eon_run_{idx:05d}"
    run_dir.mkdir(parents=True)
    halted_file = run_dir / config.halted_structure_filename
    halted_file.touch()

    result = _run_single_eon_simulation(
        idx=idx,
        structure=structure,
        potential=potential,
        config=config,
        base_workdir=tmp_path,
        physics_baseline=None
    )

    assert result is not None
    assert result.tags["provenance"] == "dynamics_halted_eon"

    # Verify subprocess called (EON binary)
    mock_subprocess_run.assert_called()

    # Verify cleanup happened (run_dir should NOT exist)
    assert not run_dir.exists()
