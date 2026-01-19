from pathlib import Path
from unittest.mock import MagicMock, patch

from mlip_autopipec.config.models import MinimalConfig, SystemConfig, TargetSystem
from mlip_autopipec.orchestration.manager import WorkflowManager
from mlip_autopipec.orchestration.models import OrchestratorConfig


def test_uat_cycle_08_end_to_end_simulation(tmp_path: Path) -> None:
    """
    UAT-08-01: End-to-End Autonomous Run Simulation

    This test simulates the full lifecycle by mocking the heavy physics engines
    but using the real WorkflowManager logic.
    """

    # 1. Setup Config
    minimal = MinimalConfig(
        project_name="uat_cycle_08",
        target_system=TargetSystem(elements=["Al"], composition={"Al": 1.0}),
        resources={"dft_code": "quantum_espresso", "parallel_cores": 4},
    )
    system_config = SystemConfig(
        minimal=minimal,
        working_dir=tmp_path,
        db_path=tmp_path / "uat.db",
        log_path=tmp_path / "uat.log",
    )
    orch_config = OrchestratorConfig(max_generations=2, workers=1)

    # 2. Mock Components
    with (
        patch("mlip_autopipec.orchestration.manager.DatabaseManager") as MockDB,
        patch("mlip_autopipec.orchestration.manager.TaskQueue"),
        patch("mlip_autopipec.orchestration.manager.Dashboard") as MockDash,
        patch("mlip_autopipec.orchestration.manager.StructureBuilder") as MockBuilder,
        patch("mlip_autopipec.orchestration.manager.SurrogatePipeline"),
        patch("mlip_autopipec.orchestration.manager.QERunner"),
        patch("mlip_autopipec.orchestration.manager.DatasetBuilder"),
        patch("mlip_autopipec.orchestration.manager.TrainConfigGenerator"),
        patch("mlip_autopipec.orchestration.manager.PacemakerWrapper"),
    ):
        # Mock behaviors
        mock_db = MockDB.return_value
        mock_db.count.return_value = 100

        mock_builder = MockBuilder.return_value
        mock_builder.build.return_value = [MagicMock()] * 10

        # 3. Initialize and Run
        manager = WorkflowManager(system_config, orch_config)
        manager.run()

        # 4. Assertions
        assert manager.state.current_generation == 2

        # Verify phases were called for each generation
        # Gen 0: Idle -> DFT -> Train -> Inference -> Loop
        # Gen 1: Idle -> DFT -> Train -> Inference -> Loop

        # Check that StructureBuilder was called twice (Gen 0 and Gen 1)
        assert MockBuilder.call_count >= 2

        # Check that Dashboard was updated
        assert MockDash.return_value.update.call_count >= 2


def test_uat_cycle_08_resume(tmp_path: Path) -> None:
    """
    UAT-08-02: Checkpoint & Resume
    """
    # 1. create a state file indicating we are in Generation 1, Phase 'training'
    state_file = tmp_path / "workflow_state.json"
    state_file.write_text('{"current_generation": 1, "status": "training", "pending_tasks": []}')

    minimal = MinimalConfig(
        project_name="uat_cycle_08",
        target_system=TargetSystem(elements=["Al"], composition={"Al": 1.0}),
        resources={"dft_code": "quantum_espresso", "parallel_cores": 4},
    )
    system_config = SystemConfig(
        minimal=minimal,
        working_dir=tmp_path,
        db_path=tmp_path / "uat.db",
        log_path=tmp_path / "uat.log",
    )
    orch_config = OrchestratorConfig(max_generations=2, workers=1)

    with (
        patch("mlip_autopipec.orchestration.manager.DatabaseManager"),
        patch("mlip_autopipec.orchestration.manager.TaskQueue"),
        patch("mlip_autopipec.orchestration.manager.Dashboard"),
        patch("mlip_autopipec.orchestration.manager.DatasetBuilder"),
        patch("mlip_autopipec.orchestration.manager.TrainConfigGenerator"),
        patch("mlip_autopipec.orchestration.manager.PacemakerWrapper") as MockPacemaker,
    ):
        manager = WorkflowManager(system_config, orch_config)

        # 2. Run
        manager.run()

        # 3. Verify it started from training phase of Gen 1
        # Should call Pacemaker (training)
        assert MockPacemaker.call_count >= 1

        # Should finish Gen 1, then do Gen 2, then stop.
        assert manager.state.current_generation == 2
