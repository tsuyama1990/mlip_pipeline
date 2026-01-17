import pytest
from pathlib import Path
from mlip_autopipec.core.workspace import WorkspaceManager
from mlip_autopipec.config.schemas.system import SystemConfig
from mlip_autopipec.config.schemas.common import MinimalConfig, TargetSystem, Composition
from mlip_autopipec.config.schemas.resources import Resources
from mlip_autopipec.exceptions import WorkspaceError

@pytest.fixture
def mock_system_config(tmp_path):
    minimal = MinimalConfig(
        project_name="TestProject",
        target_system=TargetSystem(
            elements=["Al"], composition=Composition({"Al": 1.0}), crystal_structure="fcc"
        ),
        resources=Resources(dft_code="quantum_espresso", parallel_cores=1)
    )
    return SystemConfig(
        minimal=minimal,
        working_dir=tmp_path / "work",
        db_path=tmp_path / "work/db/test.db",
        log_path=tmp_path / "work/logs/test.log"
    )

def test_setup_workspace_creates_dirs(mock_system_config):
    WorkspaceManager.setup_workspace(mock_system_config)

    assert mock_system_config.working_dir.exists()
    assert mock_system_config.db_path.parent.exists()
    assert mock_system_config.log_path.parent.exists()

def test_setup_workspace_permission_error(mock_system_config):
    # Simulate a permission error by making the parent read-only
    parent = mock_system_config.working_dir.parent
    # Ensure it exists first (it is tmp_path, so it does)

    import os
    # Make tmp_path read-only
    os.chmod(parent, 0o500)

    try:
        with pytest.raises(WorkspaceError):
            WorkspaceManager.setup_workspace(mock_system_config)
    finally:
        os.chmod(parent, 0o700)
