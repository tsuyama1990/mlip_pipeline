from pathlib import Path

import ase.db
import pytest
from mlip_autopipec.core.database import DatabaseManager

from mlip_autopipec.config.models import MinimalConfig, Resources, SystemConfig, TargetSystem


@pytest.fixture
def sample_system_config(tmp_path: Path) -> SystemConfig:
    minimal = MinimalConfig(
        project_name="TestDB",
        target_system=TargetSystem(elements=["Ag"], composition={"Ag": 1.0}),
        resources=Resources(dft_code="quantum_espresso", parallel_cores=2),
    )
    return SystemConfig(
        minimal=minimal,
        working_dir=tmp_path,
        db_path=tmp_path / "project.db",
        log_path=tmp_path / "system.log",
    )


def test_database_initialization(sample_system_config: SystemConfig) -> None:
    db_manager = DatabaseManager(sample_system_config.db_path)
    db_manager.initialize(sample_system_config)

    assert sample_system_config.db_path.exists()

    # Verify metadata
    with ase.db.connect(sample_system_config.db_path) as conn:
        # Force initialization if needed
        assert conn.count() == 0
        metadata = conn.metadata
        assert "minimal" in metadata
        assert metadata["minimal"]["project_name"] == "TestDB"


def test_database_get_metadata(sample_system_config: SystemConfig) -> None:
    db_manager = DatabaseManager(sample_system_config.db_path)
    db_manager.initialize(sample_system_config)

    metadata = db_manager.get_metadata()
    assert metadata["minimal"]["project_name"] == "TestDB"


def test_database_persistence(sample_system_config: SystemConfig) -> None:
    db_manager = DatabaseManager(sample_system_config.db_path)
    db_manager.initialize(sample_system_config)

    # Re-instantiate
    db_manager2 = DatabaseManager(sample_system_config.db_path)
    metadata = db_manager2.get_metadata()
    assert metadata["minimal"]["project_name"] == "TestDB"
