"""Unit tests for MaceManager."""

from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.config import MaceConfig


# Use a fixture to delay import until the file is created in next step
@pytest.fixture
def mace_manager_class() -> type:
    from pyacemaker.oracle.mace_manager import MaceManager

    return MaceManager


def test_mace_manager_initialization(mace_manager_class: type) -> None:
    """Test MaceManager initialization."""
    config = MaceConfig(model_path="medium", device="cpu")
    manager = mace_manager_class(config)
    assert manager.config == config


@patch("pyacemaker.oracle.mace_manager.HAS_MACE", True)
@patch("pyacemaker.oracle.mace_manager.MACECalculator")
def test_mace_manager_load_model(
    mock_calculator_cls: MagicMock, mace_manager_class: type
) -> None:
    """Test loading MACE model."""
    config = MaceConfig(model_path="medium", device="cpu")
    manager = mace_manager_class(config)

    manager.load_model()
    mock_calculator_cls.assert_called_once_with(
        model_paths="medium", device="cpu", default_dtype="float64"
    )
    assert manager.calculator is not None


@patch("pyacemaker.oracle.mace_manager.HAS_MACE", True)
@patch("pyacemaker.oracle.mace_manager.MACECalculator")
def test_mace_manager_compute(
    mock_calculator_cls: MagicMock, mace_manager_class: type
) -> None:
    """Test compute method."""
    config = MaceConfig(model_path="medium")
    manager = mace_manager_class(config)

    # Mock calculator instance
    mock_calc = MagicMock()
    mock_calculator_cls.return_value = mock_calc

    # Mock atoms
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])

    # Run
    result = manager.compute(atoms)

    # Check that calculator was assigned
    # Note: result.calc might be a property proxy in some ASE versions, but checking identity usually works
    assert result.calc == mock_calc

    # Check that calculation was triggered
    # We can check if `get_potential_energy` was called on the calculator.
    # Standard ASE calculators have `get_potential_energy`.
    mock_calc.get_potential_energy.assert_called()

    # Verify result is the same atoms object (copy)
    assert isinstance(result, Atoms)


def test_build_train_command_validation(mace_manager_class: type, tmp_path) -> None:
    """Test train command construction with parameter validation."""
    config = MaceConfig(model_path="medium")
    manager = mace_manager_class(config)

    dataset_path = tmp_path / "data.xyz"
    work_dir = tmp_path / "work"

    # Valid params
    params = {
        "max_num_epochs": 100,
        "batch_size": 32,
        "lr": 1e-3, # Scientific notation
        "loss": "energy_forces",
    }

    cmd = manager._build_train_command(dataset_path, work_dir, params)
    assert "--max_num_epochs" in cmd
    assert "100" in cmd
    assert "--lr" in cmd
    assert "0.001" in cmd # 1e-3 -> 0.001 str conversion usually

    # Invalid key
    params_invalid_key = {"bad_key": 1}
    cmd_bad = manager._build_train_command(dataset_path, work_dir, params_invalid_key)
    assert "--bad_key" not in cmd_bad

    # Invalid value (injection attempt)
    params_injection = {"model": "; rm -rf /"}
    cmd_inj = manager._build_train_command(dataset_path, work_dir, params_injection)
    # Value validation should block this
    assert "; rm -rf /" not in cmd_inj
