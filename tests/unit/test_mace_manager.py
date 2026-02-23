"""Unit tests for MaceManager."""

from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.config import MaceConfig


# Use a fixture to delay import until the file is created in next step
@pytest.fixture
def MaceManagerClass() -> type:
    from pyacemaker.oracle.mace_manager import MaceManager
    return MaceManager


def test_mace_manager_initialization(MaceManagerClass: type) -> None:
    """Test MaceManager initialization."""
    config = MaceConfig(model_path="medium", device="cpu")
    manager = MaceManagerClass(config)
    assert manager.config == config


@patch("pyacemaker.oracle.mace_manager.HAS_MACE", True)
@patch("pyacemaker.oracle.mace_manager.MACECalculator")
def test_mace_manager_load_model(mock_calculator_cls: MagicMock, MaceManagerClass: type) -> None:
    """Test loading MACE model."""
    config = MaceConfig(model_path="medium", device="cpu")
    manager = MaceManagerClass(config)

    manager.load_model()
    mock_calculator_cls.assert_called_once_with(
        model_paths="medium", device="cpu", default_dtype="float64"
    )
    assert manager.calculator is not None


@patch("pyacemaker.oracle.mace_manager.HAS_MACE", True)
@patch("pyacemaker.oracle.mace_manager.MACECalculator")
def test_mace_manager_compute(mock_calculator_cls: MagicMock, MaceManagerClass: type) -> None:
    """Test compute method."""
    config = MaceConfig(model_path="medium")
    manager = MaceManagerClass(config)

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
    # ASE get_potential_energy calls calculate() internally usually
    # But we called get_potential_energy explicitly in manager
    # Wait, mock_calc is an object.
    # ASE atoms.calc = mock_calc
    # atoms.get_potential_energy() -> calls mock_calc.get_potential_energy()
    # verify that was called.
    # But wait, in manager implementation I will call atoms.get_potential_energy()
    # So I should mock that method on the calculator.

    # Actually ASE calculator interface:
    # calculator.get_potential_energy(atoms=...)
    # But atoms.get_potential_energy() calls self.calc.get_potential_energy(self)

    # Let's verify that get_potential_energy was called on the atom
    # OR that the calculator's calculate method was called.
    # But generic ASE calculator usually has `calculate`.
    # Let's just check that get_potential_energy was called on the atom copy if we mock atoms?
    # No, we pass real Atoms object.

    # We can check if `get_potential_energy` was called on the calculator.
    # Standard ASE calculators have `get_potential_energy`.
    mock_calc.get_potential_energy.assert_called()
