# ruff: noqa: D101, D102
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms
from dask.distributed import Client, LocalCluster

from mlip_autopipec.app import expand_config, run_workflow
from mlip_autopipec.config_schemas import SystemConfig, UserConfig


@pytest.fixture
def mock_system_config(tmp_path: Path) -> SystemConfig:
    """Provide a mock SystemConfig for testing the application."""
    config_dict = {
        "target_system": {"elements": ["Cu"], "composition": {"Cu": 1.0}},
        "simulation_goal": "melt_quench",
    }
    user_config = UserConfig(**config_dict)
    return expand_config(user_config)


def test_app_with_dask_local_cluster(
    mock_system_config: SystemConfig, mocker: MagicMock, tmp_path: Path
) -> None:
    """Test the main application with a local Dask cluster."""
    # Mock the dependencies
    mock_db_manager = MagicMock()
    mock_dft_factory = MagicMock()
    mock_trainer = MagicMock()
    mock_lammps_runner = MagicMock()
    mock_lammps_runner.run.return_value = iter([])
    mocker.patch("mlip_autopipec.app.LammpsRunner", return_value=mock_lammps_runner)
    mocker.patch("mlip_autopipec.app.UncertaintyQuantifier", MagicMock())

    # Configure the mocks to return specific values
    mock_dft_factory.run.return_value = Atoms("Cu")
    mock_lammps_runner = MagicMock()
    mock_lammps_runner.run.return_value = iter(
        [(Atoms("Cu"), np.array([[1.0, 1.0, 1.0]]))]
    )
    mocker.patch("mlip_autopipec.app.LammpsRunner", return_value=mock_lammps_runner)

    with LocalCluster(n_workers=1, threads_per_worker=1) as cluster:  # type: ignore[no-untyped-call]
        with Client(cluster) as client:  # type: ignore[no-untyped-call]
            mocker.patch("mlip_autopipec.app.setup_dask_client", return_value=client)

            run_workflow(
                config=mock_system_config,
                checkpoint_path=tmp_path / "checkpoint.json",
                db_manager=mock_db_manager,
                dft_factory=mock_dft_factory,
                trainer=mock_trainer,
            )

    # Verify that the database write method was called
    mock_db_manager.write_calculation.assert_called_once()
    call_args = mock_db_manager.write_calculation.call_args
    assert isinstance(call_args.kwargs["atoms"], Atoms)
    assert "force_mask" in call_args.kwargs
