import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from ase import Atoms

from mlip_autopipec.config.models import SystemConfig, MinimalConfig, TargetSystem, Resources
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.config.schemas.surrogate import SurrogateConfig
from mlip_autopipec.config.schemas.training import TrainingConfig
from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.orchestration.manager import WorkflowManager
from mlip_autopipec.orchestration.models import OrchestratorConfig

def test_uat_08_01_end_to_end_autonomous_run(mocker: MockerFixture, tmp_path: Path) -> None:
    """
    UAT-08-01: Verify that the system can perform a complete "Zero-Human" run.
    """
    # Mock Physics Engines
    mocker.patch("mlip_autopipec.orchestration.manager.StructureBuilder")
    mocker.patch("mlip_autopipec.orchestration.manager.SurrogatePipeline")
    mocker.patch("mlip_autopipec.orchestration.manager.QERunner")
    mocker.patch("mlip_autopipec.orchestration.manager.PacemakerWrapper")
    mocker.patch("mlip_autopipec.orchestration.manager.LammpsRunner")
    mocker.patch("mlip_autopipec.orchestration.manager.DatabaseManager")
    mocker.patch("mlip_autopipec.orchestration.manager.TaskQueue")
    mocker.patch("mlip_autopipec.orchestration.manager.Dashboard")
    mocker.patch("mlip_autopipec.orchestration.manager.DatasetBuilder")
    mocker.patch("mlip_autopipec.orchestration.manager.TrainConfigGenerator")

    # Create dummy potential for inference config
    pot_path = tmp_path / "dummy.yace"
    pot_path.touch()

    # Configs
    minimal = MinimalConfig(
        project_name="uat_project",
        target_system=TargetSystem(
            name="Cu",
            structure_type="bulk",
            elements=["Cu"],
            composition={"Cu": 1.0}
        ),
        resources=Resources(dft_code="quantum_espresso", parallel_cores=4)
    )
    system_config = SystemConfig(
        minimal=minimal,
        working_dir=tmp_path,
        db_path=tmp_path / "uat.db",
        log_path=tmp_path / "uat.log",
        dft_config=DFTConfig(pseudo_dir=tmp_path / "pseudos"),
        surrogate_config=SurrogateConfig(),
        training_config=TrainingConfig(data_source_db=tmp_path / "uat.db"),
        inference_config=InferenceConfig(temperature=300.0, potential_path=pot_path),
    )
    orch_config = OrchestratorConfig(max_generations=2, workers=1)

    tmp_path.mkdir(parents=True, exist_ok=True)

    # Run
    manager = WorkflowManager(system_config, orch_config)
    manager.run()

    # Verify completion
    assert manager.state.current_generation == 2
    assert (tmp_path / "workflow_state.json").exists()


def test_uat_08_02_checkpoint_resume(mocker: MockerFixture, tmp_path: Path) -> None:
    """
    UAT-08-02: Verify Checkpoint & Resume.
    """
    # Mocks
    mocker.patch("mlip_autopipec.orchestration.manager.StructureBuilder")
    mocker.patch("mlip_autopipec.orchestration.manager.SurrogatePipeline")
    mocker.patch("mlip_autopipec.orchestration.manager.QERunner")
    mocker.patch("mlip_autopipec.orchestration.manager.PacemakerWrapper")
    mocker.patch("mlip_autopipec.orchestration.manager.LammpsRunner")
    mocker.patch("mlip_autopipec.orchestration.manager.DatabaseManager")
    mocker.patch("mlip_autopipec.orchestration.manager.TaskQueue")
    mocker.patch("mlip_autopipec.orchestration.manager.Dashboard")
    mocker.patch("mlip_autopipec.orchestration.manager.DatasetBuilder")
    mocker.patch("mlip_autopipec.orchestration.manager.TrainConfigGenerator")

    # Create valid state file for Gen 1, Status 'training'
    state_file = tmp_path / "workflow_state.json"
    state_data = {
        "current_generation": 1,
        "status": "training",
        "pending_tasks": []
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    with state_file.open("w") as f:
        json.dump(state_data, f)

    # Configs
    minimal = MinimalConfig(
        project_name="uat_resume",
        target_system=TargetSystem(
            name="Cu",
            structure_type="bulk",
            elements=["Cu"],
            composition={"Cu": 1.0}
        ),
        resources=Resources(dft_code="quantum_espresso", parallel_cores=4)
    )
    system_config = SystemConfig(
        minimal=minimal,
        working_dir=tmp_path,
        db_path=tmp_path / "uat.db",
        log_path=tmp_path / "uat.log"
    )
    orch_config = OrchestratorConfig(max_generations=3)

    # Initialize
    manager = WorkflowManager(system_config, orch_config)

    # Check loaded state
    assert manager.state.current_generation == 1
    assert manager.state.status == "training"

    # Run
    manager.run()

    # Should finish at max_generations
    assert manager.state.current_generation == 3
