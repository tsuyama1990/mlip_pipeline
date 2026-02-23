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
