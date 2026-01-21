
import pytest
from ase import Atoms

from mlip_autopipec.core.database import DatabaseManager


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"

@pytest.fixture
def db_manager(db_path):
    manager = DatabaseManager(db_path)
    manager.initialize()
    return manager

def test_database_initialize(db_path):
    manager = DatabaseManager(db_path)
    manager.initialize()
    assert db_path.exists()

def test_database_initialize_existing(db_path):
    """Test initializing an already existing database."""
    # First init
    manager = DatabaseManager(db_path)
    manager.initialize()
    assert db_path.exists()

    # Write something to ensure it's not overwritten/cleared
    manager.add_structure(Atoms("H"), {"status": "pending", "config_type": "test", "generation": 0})
    assert manager.count() == 1

    # Second init
    manager2 = DatabaseManager(db_path)
    manager2.initialize()
    assert db_path.exists()
    assert manager2.count() == 1 # Should persist data

def test_context_manager(db_path):
    """Test using DatabaseManager as a context manager."""
    with DatabaseManager(db_path) as db:
        assert db._connection is not None
        db.add_structure(Atoms("H"), {"status": "pending", "config_type": "test", "generation": 0})
        assert db.count() == 1
    # After exit, connection should be closed (set to None in our implementation)
    # Re-instantiate to check logic
    db2 = DatabaseManager(db_path)
    assert db2._connection is None

def test_add_structure(db_manager):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    metadata = {
        "status": "pending",
        "config_type": "dimer",
        "generation": 0
    }

    struct_id = db_manager.add_structure(atoms, metadata)
    assert struct_id > 0
    assert db_manager.count() == 1
    assert db_manager.count(selection="status=pending") == 1

def test_update_status(db_manager):
    atoms = Atoms("H")
    struct_id = db_manager.add_structure(atoms, {"status": "pending", "config_type": "test", "generation": 0})

    db_manager.update_status(struct_id, "running")
    assert db_manager.count(selection="status=running") == 1
    assert db_manager.count(selection="status=pending") == 0
