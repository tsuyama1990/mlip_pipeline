# ruff: noqa: D101, D102, D103, T201
"""Tests for the main application."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from ase import Atoms
import dask
from dask.distributed import Client
from typer.testing import CliRunner

from mlip_autopipec.app import app, expand_config
from mlip_autopipec.config_schemas import CalculationMetadata, SystemConfig, UserConfig


def dummy_dft_task(
    config: SystemConfig, atoms: Atoms, force_mask: np.ndarray | None
):
    """A dummy function to simulate the dft_task_wrapper, returning serializable objects."""
    print("Running dummy DFT task.")
    metadata = CalculationMetadata(stage="test", uuid="test-uuid")
    return atoms, metadata, force_mask


def test_app_with_synchronous_scheduler(tmp_path):
    """Test the main application with the synchronous Dask scheduler."""
    config_file = tmp_path / "config.yaml"
    user_config_dict = {
        "target_system": {"elements": ["Cu"], "composition": {"Cu": 1.0}},
        "simulation_goal": "melt_quench",
    }
    with open(config_file, "w") as f:
        yaml.dump(user_config_dict, f)

    user_config = UserConfig(**user_config_dict)
    system_config = expand_config(user_config)
    system_config.inference.total_simulation_steps = 5
    system_config.inference.uncertainty_threshold = 3.0

    test_atoms = Atoms("H")
    test_mask = np.array([1.0])

    with patch("mlip_autopipec.app.expand_config", return_value=system_config), \
         patch("mlip_autopipec.app.LammpsRunner") as mock_lammps_runner, \
         patch("mlip_autopipec.app.DatabaseManager") as mock_db_manager, \
         patch("mlip_autopipec.app.dft_task_wrapper", side_effect=dummy_dft_task) as mock_dft_task:

        mock_lammps_instance = mock_lammps_runner.return_value
        mock_lammps_instance.run.return_value = iter([
            (Atoms("He"), 1.0, None),
            (test_atoms, 4.5, test_mask),
            (Atoms("Li"), 2.0, None),
        ])

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--config", str(config_file), "--scheduler", "synchronous"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
