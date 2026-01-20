import pytest
from ase import Atoms

from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.exceptions import DatabaseException


def test_database_initialization(tmp_path):
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)
    db.initialize()
    assert db_path.exists()

def test_add_calculation(tmp_path):
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)
    db.initialize()

    atoms = Atoms("H2O")
    atoms.info["energy"] = -10.5

    row_id = db.add_calculation(atoms, {"calculation_type": "scf"})
    assert row_id == 1
    assert db.count() == 1

def test_get_pending_calculations(tmp_path):
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)
    db.initialize()

    atoms = Atoms("Si")
    db.save_candidate(atoms, {"status": "pending", "generation": 0})

    pending = db.get_pending_calculations()
    assert len(pending) == 1
    assert pending[0].get_chemical_symbols()[0] == "Si"

def test_database_connection_failure(tmp_path, mocker):
    db_path = tmp_path / "fail.db"
    db = DatabaseManager(db_path)

    # Mock ase.db.connect to raise Exception
    mocker.patch("ase.db.connect", side_effect=OSError("Disk full"))

    with pytest.raises(DatabaseException, match="Failed to initialize database"):
        db.initialize()

def test_write_failure(tmp_path, mocker):
    db_path = tmp_path / "write_fail.db"
    db = DatabaseManager(db_path)
    db.initialize()

    # Mock the internal writer's write_atoms method to fail
    # Since we refactored, db._writer is the target
    # But it's easier to mock the connection object itself if we can access it or if _writer uses it.
    # db._writer._connection is the ase.db object.

    # We can mock the write method of the connection object
    mocker.patch.object(db._writer._connection, "write", side_effect=Exception("DB Locked"))

    atoms = Atoms("H")
    with pytest.raises(DatabaseException, match="Failed to write atoms"):
        db.add_calculation(atoms, {})
