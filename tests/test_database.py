from uuid import uuid4

import pytest
from ase.db import connect

from mlip_autopipec.config.schemas.system import SystemConfig, WorkflowConfig
from mlip_autopipec.core.database import DatabaseManager


@pytest.fixture
def system_config(tmp_path):
    # minimal valid system config
    return SystemConfig(
        project_name="DBTest",
        run_uuid=uuid4(),
        workflow_config=WorkflowConfig(),
        db_path="test.db",
        working_dir=tmp_path
    )

def test_database_initialization(tmp_path, system_config):
    db_path = tmp_path / "test.db"
    db_path_str = str(db_path)

    # Initialize
    db_manager = DatabaseManager(db_path_str)
    db_manager.initialize(system_config)

    assert db_path.exists()

    # Check metadata
    with connect(db_path_str) as db:
        meta = db.metadata
        assert "system_config" in meta
        assert meta["system_config"]["project_name"] == "DBTest"

def test_add_structure(tmp_path, system_config):
    from ase import Atoms
    db_path = tmp_path / "structure.db"
    db_manager = DatabaseManager(str(db_path))
    db_manager.initialize(system_config)

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    meta = {"source": "test", "generation": 1}

    # Should work
    row_id = db_manager.add_structure(atoms, metadata=meta)
    assert row_id is not None

    # Verify
    with connect(str(db_path)) as db:
        row = db.get(id=row_id)
        assert row.source == "test"
