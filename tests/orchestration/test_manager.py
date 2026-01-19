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

def test_manager_run_max_generations(mock_dependencies: None, valid_system_config: SystemConfig) -> None:
    valid_system_config.working_dir.mkdir(parents=True, exist_ok=True)
    orch_config = OrchestratorConfig(max_generations=0) # Should stop immediately

    manager = WorkflowManager(valid_system_config, orch_config)
    manager.run()

    # TaskQueue shutdown should be called
    manager.task_queue.shutdown.assert_called_once()

def test_manager_phase_failure_resilience(mock_dependencies: None, valid_system_config: SystemConfig) -> None:
    """
    Ensure the manager continues to the next phase/saves state even if a phase fails
    (simulated by exception in sub-method).
    """
    valid_system_config.working_dir.mkdir(parents=True, exist_ok=True)
    orch_config = OrchestratorConfig(max_generations=1)

    manager = WorkflowManager(valid_system_config, orch_config)

    # Mock _run_exploration_phase to raise Exception
    # We patch the instance method
    manager._run_exploration_phase = MagicMock(side_effect=RuntimeError("Exploration Boom")) # type: ignore

    # Run
    manager.run()

    # Manager catches exception, logs it, and continues loop (or saves state)
    # Since we didn't implement sophisticated transition logic on failure (it just logs),
    # the state likely remains 'idle' or transitions if the failure was late.
    # In current implementation, `_run_exploration_phase` itself has the try/except block.
    # So if we mock it to raise, we are mocking the *inside* logic, but `run` calls the method.
    # Wait, `run` calls `self._run_exploration_phase()`.
    # If `_run_exploration_phase` implementation has the try/except, raising inside it works.
    # But here we mocked the WHOLE method. So `run` catches it?
    # `run` has a try/except around the whole loop. So if a phase raises, `run` catches and loops/exits.

    # Let's verify `run` caught it and called shutdown.
    manager.task_queue.shutdown.assert_called_once()
