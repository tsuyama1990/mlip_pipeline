from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mlip_autopipec.components.dynamics.lammps import LAMMPSDynamics
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

def test_lammps_dynamics_explore_schedule(
    tmp_path: Path,
    structure: Structure,
    potential: Potential
) -> None:
    config = LAMMPSDynamicsConfig(
        name=DynamicsType.LAMMPS,
        temperature=300.0,
        temperature_schedule={1: 400.0, 2: 500.0},
        max_workers=1
    )

    dynamics = LAMMPSDynamics(config)

    # Mock ProcessPoolExecutor to capture the config passed to the worker
    with patch("concurrent.futures.ProcessPoolExecutor") as mock_executor_cls:
        mock_executor = MagicMock()
        mock_executor_cls.return_value.__enter__.return_value = mock_executor

        # We need to capture calls to submit
        mock_future = MagicMock()
        mock_future.result.return_value = None # Assume no halt for simplicity
        mock_executor.submit.return_value = mock_future

        with patch("concurrent.futures.as_completed", return_value=[mock_future]):

            # Cycle 1 -> Should use 400.0
            list(dynamics.explore(potential, [structure], workdir=tmp_path, cycle=1))

            # Verify the config passed to submit has temperature=400.0
            args, _ = mock_executor.submit.call_args
            # args: fn, idx, structure, potential, config, ...
            # Index 4 is config
            passed_config = args[4]
            assert passed_config.temperature == 400.0

            # Cycle 2 -> Should use 500.0
            list(dynamics.explore(potential, [structure], workdir=tmp_path, cycle=2))
            args, _ = mock_executor.submit.call_args
            passed_config = args[4]
            assert passed_config.temperature == 500.0

            # Cycle 3 -> No schedule, use default 300.0 (or base config)
            list(dynamics.explore(potential, [structure], workdir=tmp_path, cycle=3))
            args, _ = mock_executor.submit.call_args
            passed_config = args[4]
            assert passed_config.temperature == 300.0
