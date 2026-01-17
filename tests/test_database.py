from pathlib import Path
from uuid import uuid4

import pytest
from ase import Atoms

from mlip_autopipec.config.models import CalculationMetadata, SystemConfig
from mlip_autopipec.core.database import DatabaseManager


@pytest.fixture
def minimal_system_config(tmp_path):
    return SystemConfig(
        project_name="TestProject",
        run_uuid=uuid4(),
        working_dir=tmp_path,
        db_path="test.db",
    )

def test_database_initialization(tmp_path: Path) -> None:
    """Test that the database can be initialized."""
    db_path = tmp_path / "test.db"
    db_manager = DatabaseManager(db_path)
    conn = db_manager.connect()
    # ase.db.connect might not create the file immediately if nothing is written,
    # or it might create an empty file.
    # In sqlite, connecting usually creates the file.
    # However, depending on ASE version/implementation, it might delay creation.
    # Let's write something to ensure it exists.
    conn.write(Atoms("H"), name="init_test")
    assert conn is not None
    assert db_path.exists()

def test_database_initialize_metadata(tmp_path: Path, minimal_system_config: SystemConfig) -> None:
    """Test that initialize stores the system config in metadata."""
    db_path = tmp_path / "metadata_test.db"
    db_manager = DatabaseManager(db_path)

    db_manager.initialize(minimal_system_config)

    assert db_path.exists()

    # Re-connect and check metadata
    conn = db_manager.connect()
    # Force metadata load if needed
    try:
        metadata = conn.metadata
    except Exception:
        pytest.fail("Failed to retrieve metadata from initialized database.")

    assert "system_config" in metadata
    stored_config = metadata["system_config"]
    assert stored_config["project_name"] == "TestProject"
    assert stored_config["run_uuid"] == str(minimal_system_config.run_uuid)


def test_write_calculation(tmp_path: Path) -> None:
    """Test writing a calculation to the database."""
    db_path = tmp_path / "test.db"
    db_manager = DatabaseManager(db_path)

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    metadata = CalculationMetadata(
        stage="test",
        uuid="uuid-12345",
    )

    db_manager.write_calculation(atoms, metadata)

    conn = db_manager.connect()
    row = conn.get(id=1)
    assert row.mlip_stage == "test"
    assert row.mlip_uuid == "uuid-12345"
    assert row.natoms == 2

def test_get_completed_calculations(tmp_path: Path) -> None:
    """Test retrieving completed calculations."""
    db_path = tmp_path / "test.db"
    db_manager = DatabaseManager(db_path)

    atoms = Atoms("He")
    metadata = CalculationMetadata(stage="test", uuid="uuid-1")
    # Write one without results (not calculated)
    db_manager.write_calculation(atoms, metadata)

    # Write another one with results (calculated)
    atoms.calc = None # clear previous if any
    from ase.calculators.singlepoint import SinglePointCalculator
    atoms2 = Atoms("He")
    atoms2.calc = SinglePointCalculator(atoms2, energy=-1.0, forces=[[0,0,0]]) # type: ignore[no-untyped-call]
    db_manager.write_calculation(atoms2, CalculationMetadata(stage="test", uuid="uuid-2"))

    # Currently get_completed_calculations uses `select(calculated=True)`.
    # SinglePointCalculator usually satisfies this if energy/forces are present.

    completed = db_manager.get_completed_calculations()
    assert len(completed) == 1
    # Verify we got the one with results (cannot check uuid easily on atoms unless we preserved it in atoms info/arrays)
    # But checking count is enough for now.
