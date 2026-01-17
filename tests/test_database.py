import pytest
import ase.db
from pathlib import Path
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.config.schemas.system import SystemConfig
from mlip_autopipec.config.schemas.common import MinimalConfig, TargetSystem, Composition
from mlip_autopipec.config.schemas.resources import Resources

@pytest.fixture
def sample_system_config(tmp_path):
    minimal = MinimalConfig(
        project_name="TestProject",
        target_system=TargetSystem(
            elements=["Al"], composition=Composition({"Al": 1.0}), crystal_structure="fcc"
        ),
        resources=Resources(dft_code="quantum_espresso", parallel_cores=1)
    )
    return SystemConfig(
        minimal=minimal,
        working_dir=tmp_path,
        db_path=tmp_path / "test.db",
        log_path=tmp_path / "test.log"
    )

def test_database_initialization(sample_system_config):
    db_path = sample_system_config.db_path
    manager = DatabaseManager(db_path)

    manager.initialize(sample_system_config)

    assert db_path.exists()

    # Verify metadata
    with ase.db.connect(db_path) as db:
        metadata = db.metadata
        assert metadata['minimal']['project_name'] == "TestProject"
        assert metadata['working_dir'] == str(sample_system_config.working_dir)

def test_get_metadata(sample_system_config):
    db_path = sample_system_config.db_path
    manager = DatabaseManager(db_path)
    manager.initialize(sample_system_config)

    meta = manager.get_metadata()
    assert meta['minimal']['resources']['dft_code'] == "quantum_espresso"

def test_add_structure(sample_system_config):
    db_path = sample_system_config.db_path
    manager = DatabaseManager(db_path)
    manager.initialize(sample_system_config)

    from ase import Atoms
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])

    row_id = manager.add_structure(atoms, data={"source": "test"})
    assert row_id == 1

    with ase.db.connect(db_path) as db:
        row = db.get(id=row_id)
        assert row.data.source == "test"
        assert len(row.toatoms()) == 2
