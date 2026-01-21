from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.core.database import DatabaseManager


def test_database_creation(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)
    assert db_path.exists()
    assert db.count() == 0

def test_add_calculation(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)

    # Create atoms with results
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    energy = -30.0
    forces = np.array([[0, 0, 0.1], [0, 0, -0.1]])
    stress = np.zeros(6)

    calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
    atoms.calc = calc

    metadata = {"config_type": "scf", "project": "test"}
    row_id = db.add_calculation(atoms, metadata)

    assert row_id == 1
    assert db.count() == 1
    assert db.count(config_type="scf") == 1

    # Verify retrieval
    # Access internal connection for testing retrieval directly if needed, or re-open
    # DatabaseManager doesn't have get() yet, but we can verify via count or generic ase.db use
    # Using internal _connection for verification
    row = db._connection.get(id=row_id)
    retrieved_atoms = row.toatoms()

    # Check properties
    assert np.allclose(retrieved_atoms.get_potential_energy(), energy)
    assert np.allclose(retrieved_atoms.get_forces(), forces)
    assert row.config_type == "scf"

def test_add_calculation_validation(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)

    atoms = Atoms("H")
    # No calculator

    with pytest.raises(ValueError, match="Atoms object must have energy"):
        db.add_calculation(atoms, {})

def test_save_and_get_pending(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)

    atoms = Atoms("Si")
    db.save_candidate(atoms, {"source": "random"})

    assert db.count(status="pending") == 1

    pending = db.get_pending_calculations()
    assert len(pending) == 1
    assert pending[0].get_chemical_formula() == "Si"
