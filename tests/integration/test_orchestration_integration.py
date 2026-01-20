from pathlib import Path
from unittest.mock import MagicMock

import pytest
from ase import Atoms
from pytest_mock import MockerFixture

from mlip_autopipec.config.models import MinimalConfig, Resources, SystemConfig, TargetSystem
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.config.schemas.surrogate import SurrogateConfig
from mlip_autopipec.config.schemas.training import TrainingConfig
from mlip_autopipec.orchestration.manager import WorkflowManager
from mlip_autopipec.orchestration.models import OrchestratorConfig


@pytest.fixture
def valid_system_config(tmp_path: Path) -> SystemConfig:
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

    # Create dummy potential for inference config validation
    pot_path = tmp_path / "dummy.yace"
    pot_path.touch()

    return SystemConfig(
        minimal=minimal,
        working_dir=tmp_path,
        db_path=tmp_path / "db.db",
        log_path=tmp_path / "log.log",
        dft_config=DFTConfig(pseudo_dir=tmp_path / "pseudos"),
        surrogate_config=SurrogateConfig(),
        training_config=TrainingConfig(data_source_db=tmp_path / "db.db"),
        inference_config=InferenceConfig(temperature=300.0, potential_path=pot_path),
    )


def test_grand_mock_workflow(mocker: MockerFixture, valid_system_config: SystemConfig) -> None:
    # 1. Mock External Components
    mock_builder = mocker.patch("mlip_autopipec.orchestration.manager.StructureBuilder")
    mock_surrogate = mocker.patch("mlip_autopipec.orchestration.manager.SurrogatePipeline")
    # We patch QERunner and LammpsRunner but don't strictly need the return object variable if not asserting methods on it directly
    mocker.patch("mlip_autopipec.orchestration.manager.QERunner")
    mock_pacemaker = mocker.patch("mlip_autopipec.orchestration.manager.PacemakerWrapper")
    mocker.patch("mlip_autopipec.orchestration.manager.LammpsRunner")
    mock_db = mocker.patch("mlip_autopipec.orchestration.manager.DatabaseManager")
    mock_queue = mocker.patch("mlip_autopipec.orchestration.manager.TaskQueue")
    mock_dashboard = mocker.patch("mlip_autopipec.orchestration.manager.Dashboard")

    # Mock config generator and dataset builder
    mocker.patch("mlip_autopipec.orchestration.manager.DatasetBuilder")
    mocker.patch("mlip_autopipec.orchestration.manager.TrainConfigGenerator")

    # 2. Setup Mock Returns
    # Generator
    dummy_atoms = Atoms("Al", positions=[[0, 0, 0]])
    mock_builder.return_value.build.return_value = [dummy_atoms]

    # Surrogate
    mock_surrogate.return_value.run.return_value = ([dummy_atoms], [])

    # DB: get_atoms must return list
    mock_db.return_value.get_atoms.return_value = [dummy_atoms]

    # TaskQueue: submit and wait
    # We need wait_for_completion to return a list of non-None results for DBT logic
    mock_res = MagicMock() # DFTResult mock
    mock_queue.return_value.wait_for_completion.return_value = [mock_res]

    # 3. Setup Orchestrator Config (1 Generation)
    orch_config = OrchestratorConfig(max_generations=1, workers=1)

    # 4. Initialize Manager
    valid_system_config.working_dir.mkdir(parents=True, exist_ok=True)
    manager = WorkflowManager(valid_system_config, orch_config)

    # 5. Run
    manager.run()

    # 6. Verify Transitions
    # Phase A
    mock_builder.assert_called()
    mock_db.return_value.save_candidate.assert_called()

    # Phase B (DFT)
    mock_db.return_value.get_atoms.assert_called_with("status=pending")
    mock_queue.return_value.submit_dft_batch.assert_called()
    mock_db.return_value.save_dft_result.assert_called()

    # Phase C (Training) - Verify it was called
    mock_pacemaker.return_value.train.assert_called()

    # Verify Dashboard updates
    assert mock_dashboard.return_value.update.call_count >= 1

    # Verify State
    assert manager.state.current_generation == 1
    assert manager.state.status == "idle"
