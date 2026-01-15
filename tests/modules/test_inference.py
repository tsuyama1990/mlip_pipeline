# ruff: noqa: D101, D102
from unittest.mock import MagicMock

import pytest
from ase import Atoms

from mlip_autopipec.config_schemas import SystemConfig
from mlip_autopipec.modules.inference import LammpsRunner


@pytest.fixture
def mock_system_config() -> SystemConfig:
    """Provides a mock SystemConfig for testing the LammpsRunner."""
    config_dict = {
        "dft": {
            "executable": {"command": "pw.x"},
            "input": {"pseudopotentials": {"Cu": "Cu.UPF"}},
        },
        "inference": {
            "uncertainty_threshold": 4.0,
            "total_simulation_steps": 10,
        },
    }
    return SystemConfig(**config_dict)


def test_lammps_runner_initialization(mock_system_config: SystemConfig):
    """Test that the LammpsRunner initializes correctly."""
    runner = LammpsRunner(config=mock_system_config, potential_path="test.yace")
    assert runner.config == mock_system_config
    assert runner.potential_path == "test.yace"


def test_runner_yields_on_uncertainty(mock_system_config: SystemConfig):
    """Verify the core generator behavior of the LammpsRunner.

    This test ensures that the runner's `run` method yields the string "stable"
    when uncertainty is low and yields an ASE `Atoms` object when the
    uncertainty threshold is exceeded.
    """
    runner = LammpsRunner(config=mock_system_config, potential_path="test.yace")
    # The mock sequence is [1.0, 1.5, 2.0, 4.5, 2.5] and threshold is 4.0
    runner._mock_uncertainty_sequence = [1.0, 1.5, 2.0, 4.5, 2.5]
    generator = runner.run()

    # The first three steps should be stable
    assert next(generator) == "stable"
    assert next(generator) == "stable"
    assert next(generator) == "stable"

    # The fourth step should exceed the threshold and yield an Atoms object
    uncertain_structure = next(generator)
    assert isinstance(uncertain_structure, Atoms)

    # After yielding the structure, the simulation should resume and be stable
    assert next(generator) == "stable"


def test_runner_stops_at_total_steps(mock_system_config: SystemConfig):
    """Test that the simulation stops after the specified number of steps."""
    mock_system_config.inference.total_simulation_steps = 5
    runner = LammpsRunner(config=mock_system_config, potential_path="test.yace")
    runner._mock_uncertainty_sequence = [1.0] * 10  # Ensure it's always stable
    generator = runner.run()

    # Consume all steps from the generator
    for _ in range(5):
        next(generator)

    # The next call should raise StopIteration
    with pytest.raises(StopIteration):
        next(generator)
