from pathlib import Path
from typing import Any
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
    # Use context manager for Executor mocking
    with patch("concurrent.futures.ProcessPoolExecutor") as mock_executor_cls:
        mock_executor = mock_executor_cls.return_value
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None

        mock_future = MagicMock()

        # Prepare result structure
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
            assert results[0].tags["provenance"] == "dynamics_halted_eon"

            # Verify internal calls via executor submission
            assert mock_executor.submit.call_count == 1
            mock_executor_cls.assert_called_with(max_workers=1)
