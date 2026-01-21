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


def test_add_structure(db_manager):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    metadata = {"status": "pending", "config_type": "dimer", "generation": 0}

    # Assuming the API returns the ID or we can count
    db_manager.add_structure(atoms, metadata)
    assert db_manager.count() == 1
    assert db_manager.count(selection="status=pending") == 1


def test_update_status(db_manager):
    atoms = Atoms("H")
    db_manager.add_structure(atoms, {"status": "pending"})

    # We need to know the ID. Typically ASE DB IDs start at 1.
    # We should verify if add_structure returns ID.
    # Spec says: returns the integer ID.

    # Let's assume implementation will handle it.
    # For TDD, I'll update the test when I implement the logic, or assume ID=1.
    db_manager.update_status(1, "running")
    assert db_manager.count(selection="status=running") == 1
    assert db_manager.count(selection="status=pending") == 0
