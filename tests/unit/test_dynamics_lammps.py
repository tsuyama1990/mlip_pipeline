from pathlib import Path
from typing import Any
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
    p = tmp_path / "test.yace"
    p.touch()
    return Potential(path=p, species=["Cu"], format="yace")


@pytest.fixture
def config() -> LAMMPSDynamicsConfig:
    return LAMMPSDynamicsConfig(
        name=DynamicsType.LAMMPS,
        n_steps=1000,
        timestep=0.002,
        uncertainty_threshold=1.5,
        max_workers=2,
    )


def test_lammps_driver_write_input(
    tmp_path: Path, structure: Structure, potential: Potential, config: LAMMPSDynamicsConfig
) -> None:
    driver = LAMMPSDriver(workdir=tmp_path, config=config)
    driver.write_input_files(structure, potential)

    # Check data.lammps
    data_file = tmp_path / config.data_filename
    assert data_file.exists()
    content_data = data_file.read_text()
    assert "2 atoms" in content_data
    assert "1 atom types" in content_data
    assert "xlo xhi" in content_data
    assert "0.0" in content_data

    # Check in.lammps
    input_file = tmp_path / config.input_filename
    assert input_file.exists()
    content_in = input_file.read_text()

    # Rigorous checks
    assert "pair_style pace" in content_in
    assert f"pair_coeff * * pace {potential.path} Cu" in content_in
    assert f"timestep        {config.timestep}" in content_in
    assert f"thermo          {config.thermo_freq}" in content_in
    assert "compute         pace_gamma all pace" in content_in
    assert "variable        max_gamma equal max(c_pace_gamma)" in content_in
    assert f"fix halt_otf all halt 10 v_max_gamma > {config.uncertainty_threshold} error hard" in content_in
    assert f"dump            1 all custom {config.thermo_freq} {config.dump_filename} id type x y z fx fy fz" in content_in
    assert f"run             {config.n_steps}" in content_in


def test_lammps_driver_parse_log_halted(tmp_path: Path, config: LAMMPSDynamicsConfig) -> None:
    # Use realistic LAMMPS log output format
    log_content = """
LAMMPS (29 Oct 2020)
...
Step Temp PotEng max_gamma
       0    300.0   -6.00000      0.10000
      10    300.0   -6.00000      1.60000
ERROR: Halt: max_gamma > 1.5 (src/fix_halt.cpp:50)
Last command: run 1000
"""
    (tmp_path / config.log_filename).write_text(log_content)

    driver = LAMMPSDriver(workdir=tmp_path, config=config)
    result = driver.parse_log()
    assert result["halted"] is True
    assert result["final_step"] == 10

    # Verify parsing handles standard integer format
    # The parsing logic splits lines and looks for "Halt".
    # And then looks for last integer-starting line.


def test_lammps_driver_parse_log_finished(tmp_path: Path, config: LAMMPSDynamicsConfig) -> None:
    log_content = """
LAMMPS (29 Oct 2020)
...
Step Temp PotEng max_gamma
       0    300.0   -6.00000      0.10000
    1000    300.0   -6.00000      0.20000
Loop time of 1.23 on 1 procs for 1000 steps with 2 atoms
"""
    (tmp_path / config.log_filename).write_text(log_content)

    driver = LAMMPSDriver(workdir=tmp_path, config=config)
    result = driver.parse_log()
    assert result["halted"] is False
    assert result["final_step"] == 1000


@patch("mlip_autopipec.components.dynamics.lammps._run_single_lammps_simulation")
def test_lammps_dynamics_explore_concurrent(
    mock_run_single: MagicMock,
    tmp_path: Path,
    structure: Structure,
    potential: Potential,
    config: LAMMPSDynamicsConfig
) -> None:
    # Use ProcessPoolExecutor mock to avoid actual multiprocessing during test
    # But since we patched the target function, real executor just calls it?
    # No, ProcessPoolExecutor pickles the function.
    # It's better to patch concurrent.futures.ProcessPoolExecutor to run synchronously or mock it.

    # Let's mock ProcessPoolExecutor to behave like a context manager returning a mock executor
    with patch("concurrent.futures.ProcessPoolExecutor") as mock_executor_cls:
        mock_executor = mock_executor_cls.return_value
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None

        # Mock submit to return a Future-like object
        mock_future = MagicMock()

        # Prepare result structure
        dump_struct = structure.model_deep_copy()
        dump_struct.uncertainty = 100.0
        dump_struct.tags["provenance"] = "dynamics_halted"

        mock_future.result.return_value = dump_struct

        # as_completed yields futures
        def side_effect_as_completed(fs: list[Any]) -> list[Any]:
            return fs

        with patch("concurrent.futures.as_completed", side_effect=side_effect_as_completed):
            mock_executor.submit.return_value = mock_future

            dynamics = LAMMPSDynamics(config)

            # Explore with 2 structures
            results = list(dynamics.explore(potential, [structure, structure], workdir=tmp_path))

            assert len(results) == 2
            assert results[0].uncertainty == 100.0

            assert mock_executor.submit.call_count == 2
            # Verify max_workers
            mock_executor_cls.assert_called_with(max_workers=2)
