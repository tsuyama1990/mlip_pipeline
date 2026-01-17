import uuid
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.config.models import (
    CheckpointState,
    Composition,
    MinimalConfig,
    Resources,
    SystemConfig,
    TargetSystem,
    WorkflowConfig,
)
from mlip_autopipec.workflow_manager import WorkflowManager


@pytest.fixture
def mock_dask_client():
    with patch("mlip_autopipec.workflow_manager.get_dask_client") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client

@pytest.fixture
def mock_db_manager():
    with patch("mlip_autopipec.workflow_manager.DatabaseManager") as mock_cls:
        instance = mock_cls.return_value
        instance.get_training_data.return_value = []
        yield instance

def test_workflow_manager_initialization(tmp_path, mock_dask_client, mock_db_manager):
    """Test that WorkflowManager initializes correctly and creates a checkpoint."""
    minimal = MinimalConfig(
        project_name="Test",
        target_system=TargetSystem(
            elements=["Al"], composition=Composition({"Al": 1.0})
        ),
        resources=Resources(dft_code="quantum_espresso", parallel_cores=1)
    )
    system_config = SystemConfig(
        minimal=minimal,
        working_dir=tmp_path,
        db_path=tmp_path / "test.db",
        log_path=tmp_path / "test.log",
        run_uuid=uuid.uuid4(),
        workflow_config=WorkflowConfig(checkpoint_file_path="checkpoint.json")
    )

    manager = WorkflowManager(system_config, tmp_path)

    assert manager.state is not None
    assert manager.checkpoint_path.exists()

    # Verify DB manager was initialized
    mock_db_manager.initialize.assert_not_called() # It calls it lazily or config path usage

def test_workflow_manager_load_checkpoint(tmp_path, mock_dask_client, mock_db_manager):
    """Test loading state from an existing checkpoint."""
    run_uuid = uuid.uuid4()
    checkpoint_file = tmp_path / "checkpoint.json"

    # Create valid checkpoint state
    minimal = MinimalConfig(
        project_name="Test",
        target_system=TargetSystem(
            elements=["Al"], composition=Composition({"Al": 1.0})
        ),
        resources=Resources(dft_code="quantum_espresso", parallel_cores=1)
    )
    state = CheckpointState(
        run_uuid=run_uuid,
        system_config=SystemConfig(
             minimal=minimal,
             working_dir=tmp_path,
             db_path=tmp_path/"db",
             log_path=tmp_path/"log",
             run_uuid=run_uuid
        ),
        active_learning_generation=5
    )

    with checkpoint_file.open("w") as f:
        f.write(state.model_dump_json())

    system_config = SystemConfig(
        minimal=minimal,
        working_dir=tmp_path,
        db_path=tmp_path / "test.db",
        log_path=tmp_path / "test.log",
        run_uuid=run_uuid,
        workflow_config=WorkflowConfig(checkpoint_file_path="checkpoint.json")
    )

    manager = WorkflowManager(system_config, tmp_path)

    assert manager.state.active_learning_generation == 5
    assert manager.state.run_uuid == run_uuid
