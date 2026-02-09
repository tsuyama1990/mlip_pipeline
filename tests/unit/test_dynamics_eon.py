from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mlip_autopipec.components.dynamics.eon import (
    EONDriver,
    EONDynamics,
    _run_single_eon_simulation,
)
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
    assert "1.0e+13" in content or "1e+13" in content
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

    # Verify imports
    assert "import sys" in content
    assert "import numpy as np" in content
    assert "from ase.io import read" in content


@patch("mlip_autopipec.components.dynamics.eon.subprocess.Popen")
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


@patch("mlip_autopipec.components.dynamics.eon.subprocess.Popen")
def test_run_single_eon_simulation(
    mock_subprocess_run: MagicMock,
    tmp_path: Path,
    structure: Structure,
    potential: Potential,
    config: EONDynamicsConfig,
) -> None:
    # Prepare environment
    idx = 0
    run_dir = tmp_path / f"eon_run_{idx:05d}"

    # Mock subprocess to simulate EON run
    # Popen returns a process object
    mock_process = MagicMock()
    mock_process.stdout = MagicMock()
    # Mock iterator behavior of stdout reading: yield one chunk then empty
    mock_process.stdout.read.side_effect = [b"mock log output", b""]
    mock_process.wait.return_value = 0
    mock_subprocess_run.return_value = mock_process

    # We need to create the halted file that EON would produce
    # Instead of mocking read(), we write a real file so read() works
    # This verifies the integration of file writing and reading

    # Create the directory structure that write_input_files would create
    # But since we are calling _run_single_eon_simulation, it calls write_input_files
    # which creates the dir. But halted file is created by EON execution (mocked).
    # So we need to create the halted file via a side effect of subprocess or just pre-create it?
    # _run_single_eon_simulation -> driver.write_input_files -> driver.run_kmc -> read halted

    # We can patch driver.run_kmc to write the halted file?
    # Or rely on the fact that run_kmc is mocked (via subprocess) and we can pre-seed the file.
    # But run_dir is created inside the function. We can't pre-seed it easily unless we know path.
    # Wait, base_workdir is passed.

    # Pre-create directory to seed halted file?
    # But write_input_files might fail if dir exists? No, mkdir(exist_ok=True).

    run_dir.mkdir(parents=True, exist_ok=True)
    halted_file = run_dir / config.halted_structure_filename

    # Write a valid XYZ/EXTXYZ content
    halted_file.write_text(
        '1\nProperties=species:S:1:pos:R:3:uncertainty:R:1 pbc="T T T" Lattice="4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0"\n'
        'Cu 0.1 0.1 0.1 100.0\n'
    )

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
    assert result.uncertainty == 100.0
    # Check positions from file
    assert np.allclose(result.positions, [[0.1, 0.1, 0.1]])

    # Verify subprocess called (EON binary)
    mock_subprocess_run.assert_called()

    # Verify output stream reading
    assert mock_process.stdout.read.call_count >= 2

    # Verify cleanup happened (run_dir should NOT exist)
    assert not run_dir.exists()
