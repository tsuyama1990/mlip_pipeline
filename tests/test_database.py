from pathlib import Path
from uuid import uuid4

from ase import Atoms

from mlip_autopipec.config.models import Resources, SystemConfig, UserInputConfig
from mlip_autopipec.core.database import DatabaseManager


def get_dummy_config(tmp_path):
    resources = Resources(dft_code="quantum_espresso", parallel_cores=4)
    user = UserInputConfig(
        project_name="test",
        target_system={
            "elements": ["Fe"],
            "composition": {"Fe": 1.0},
            "crystal_structure": "bcc"
        },
        simulation_goal={"type": "melt_quench"},
        resources=resources
    )
    return SystemConfig(
        user_input=user,
        working_dir=tmp_path,
        db_path=tmp_path / "test.db",
        log_path=tmp_path / "system.log",
        run_uuid=uuid4()
    )

def test_database_initialization_metadata(tmp_path: Path) -> None:
    """Test that the database initializes with SystemConfig metadata."""
    config = get_dummy_config(tmp_path)
    db_manager = DatabaseManager(config.db_path)

    db_manager.initialize(config)

    assert config.db_path.exists()

    # Read back metadata
    metadata = db_manager.get_metadata()
    # Assuming it returns the stored dict

    # Note: SystemConfig serialized usually has 'user_input' etc.
    # We check if project_name from user_input is present if flattened, or verify structure
    # ASE DB metadata is a dict.

    # If implementation stores model_dump(), we expect keys like 'user_input', 'run_uuid'.
    assert "user_input" in metadata
    assert metadata["user_input"]["project_name"] == "test"

def test_add_structure(tmp_path: Path) -> None:
    """Test adding structure with metadata."""
    config = get_dummy_config(tmp_path)
    db_manager = DatabaseManager(config.db_path)
    db_manager.initialize(config)

    atoms = Atoms("Fe")
    meta = {"source": "test", "gen": 0}

    db_manager.add_structure(atoms, **meta)

    from ase.db import connect
    with connect(config.db_path) as conn:
        row = conn.get(id=1)
        assert row.source == "test"
        assert row.gen == 0
