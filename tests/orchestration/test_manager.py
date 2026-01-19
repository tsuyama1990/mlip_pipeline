import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from mlip_autopipec.config.models import SystemConfig, MinimalConfig, TargetSystem, Resources
from mlip_autopipec.orchestration.manager import WorkflowManager
from mlip_autopipec.orchestration.models import OrchestratorConfig, WorkflowState


@pytest.fixture
def mock_dependencies(mocker: MockerFixture) -> None:
    mocker.patch("mlip_autopipec.orchestration.manager.TaskQueue")
    mocker.patch("mlip_autopipec.orchestration.manager.DatabaseManager")
    mocker.patch("mlip_autopipec.orchestration.manager.Dashboard")


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


@pytest.fixture
def valid_system_config(temp_workspace: Path) -> SystemConfig:
    minimal = MinimalConfig(
        project_name="test_project",
        target_system=TargetSystem(
            name="Al",
            structure_type="bulk",
            elements=["Al"],
            composition={"Al": 1.0}
        ),
        resources=Resources(dft_code="quantum_espresso", parallel_cores=4)
    )
    return SystemConfig(
        minimal=minimal,
        working_dir=temp_workspace,
        db_path=temp_workspace / "db.db",
        log_path=temp_workspace / "log.log"
    )


def test_manager_init(mock_dependencies: None, valid_system_config: SystemConfig) -> None:
    orch_config = OrchestratorConfig()

    # Ensure working dir exists (SystemConfig assumes factory creates it, but here we mock)
    valid_system_config.working_dir.mkdir(parents=True, exist_ok=True)

    manager = WorkflowManager(valid_system_config, orch_config)

    assert manager.state.current_generation == 0
    assert manager.state.status == "idle"
    assert manager.state_file == valid_system_config.working_dir / "workflow_state.json"


def test_manager_save_state(mock_dependencies: None, valid_system_config: SystemConfig) -> None:
    valid_system_config.working_dir.mkdir(parents=True, exist_ok=True)
    orch_config = OrchestratorConfig()
    manager = WorkflowManager(valid_system_config, orch_config)

    manager.state.current_generation = 2
    manager.state.status = "training"
    manager._save_state()  # Accessing protected method for test

    assert manager.state_file.exists()
    with manager.state_file.open("r") as f:
        data = json.load(f)
        assert data["current_generation"] == 2
        assert data["status"] == "training"


def test_manager_resume_state(mock_dependencies: None, valid_system_config: SystemConfig) -> None:
    # Pre-create state file
    valid_system_config.working_dir.mkdir(parents=True, exist_ok=True)
    state_file = valid_system_config.working_dir / "workflow_state.json"
    initial_state = WorkflowState(current_generation=3, status="inference")
    with state_file.open("w") as f:
        f.write(initial_state.model_dump_json())

    orch_config = OrchestratorConfig()

    manager = WorkflowManager(valid_system_config, orch_config)

    assert manager.state.current_generation == 3
    assert manager.state.status == "inference"
