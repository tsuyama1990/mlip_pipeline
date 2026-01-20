"""
Tests for DatabaseManager.
"""
import pytest
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.core.exceptions import DFTError

def test_database_add_calculation(tmp_path):
    """Test adding a calculation to the database."""
    db_path = tmp_path / "test.db"
    manager = DatabaseManager(db_path)

    atoms = Atoms("Si2", positions=[[0,0,0], [1.1,1.1,1.1]])
    energy = -10.5
    forces = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    stress = np.zeros(6)

    calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
    atoms.calc = calc

    metadata = {"config_type": "test", "extra": 123}

    try:
        row_id = manager.add_calculation(atoms, metadata)
    except TypeError as e:
        print(f"Caught TypeError: {e}")
        raise

    assert row_id == 1

    # Verify count
    assert manager.count(config_type="test") == 1
    assert manager.count(extra=123) == 1

def test_database_missing_properties(tmp_path):
    """Test handling of missing properties."""
    db_path = tmp_path / "test.db"
    manager = DatabaseManager(db_path)

    atoms = Atoms("H")
    # No calculator

    with pytest.raises(DFTError, match="Atoms object missing energy or forces"):
        manager.add_calculation(atoms, {})

def test_database_nan_check(tmp_path):
    """Test NaN validation."""
    db_path = tmp_path / "test.db"
    manager = DatabaseManager(db_path)

    atoms = Atoms("H")
    calc = SinglePointCalculator(atoms, energy=np.nan, forces=[[0,0,0]])
    atoms.calc = calc

    with pytest.raises(DFTError, match="Energy is NaN or Inf"):
        manager.add_calculation(atoms, {})
