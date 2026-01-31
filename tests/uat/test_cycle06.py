import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from mlip_autopipec.domain_models.config import (
    Config, OrchestratorConfig, LoggingConfig, PotentialConfig,
    BulkStructureGenConfig, MDConfig, DFTConfig, TrainingConfig, ValidationConfig
)
from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState
from mlip_autopipec.orchestration.manager import WorkflowManager

@pytest.fixture
def valid_config(tmp_path):
    return Config(
        project_name="UAT_Cycle06",
        logging=LoggingConfig(level="INFO", file_path=tmp_path / "log.txt"),
        orchestrator=OrchestratorConfig(max_iterations=1, uncertainty_threshold=0.1, validation_frequency=1),
        potential=PotentialConfig(elements=["Si"], cutoff=5.0, seed=42, pair_style="hybrid/overlay"),
        structure_gen=BulkStructureGenConfig(strategy="bulk", element="Si", crystal_structure="diamond", lattice_constant=5.43, supercell=(1,1,1)),
        md=MDConfig(temperature=300, n_steps=100, ensemble="NVT", timestep=0.001),
        dft=DFTConfig(command="pw.x", pseudopotentials={"Si": "Si.upf"}, ecutwfc=40.0, kspacing=0.04),
        training=TrainingConfig(batch_size=10, max_epochs=1),
        validation=ValidationConfig(report_path=tmp_path/"report.html")
    )

# UAT Scenario 6.1: The Autonomous Loop
def test_uat_autonomous_loop(valid_config, tmp_path):

    # We mock the phases to simulate the loop without running real physics
    with patch("mlip_autopipec.orchestration.manager.StateManager") as MockStateMgr, \
         patch.object(WorkflowManager, 'explore', return_value=True), \
         patch.object(WorkflowManager, 'select', return_value=[MagicMock()]), \
         patch.object(WorkflowManager, 'calculate', return_value=True), \
         patch.object(WorkflowManager, 'train', return_value=Path("new.yace")), \
         patch.object(WorkflowManager, 'validate', return_value=True):

        # Initial State
        state = WorkflowState(
            project_name="UAT_Cycle06",
            dataset_path=tmp_path / "data.pckl",
            current_phase=WorkflowPhase.EXPLORATION,
            generation=0
        )
        MockStateMgr.return_value.load.return_value = state

        manager = WorkflowManager(valid_config, work_dir=tmp_path)

        # Run loop (simulated by calling step repeatedly)
        # Gen 0: EXPLORATION -> SELECTION
        manager.step()
        assert manager.state.current_phase == WorkflowPhase.SELECTION

        # Gen 0: SELECTION -> CALCULATION
        manager.step()
        assert manager.state.current_phase == WorkflowPhase.CALCULATION

        # Gen 0: CALCULATION -> TRAINING
        manager.step()
        assert manager.state.current_phase == WorkflowPhase.TRAINING

        # Gen 0: TRAINING -> VALIDATION
        manager.step()
        assert manager.state.current_phase == WorkflowPhase.VALIDATION

        # Gen 0: VALIDATION -> Gen 1 EXPLORATION
        manager.step()
        assert manager.state.generation == 1
        assert manager.state.current_phase == WorkflowPhase.EXPLORATION

        # Check max iterations reached
        assert manager.state.generation >= valid_config.orchestrator.max_iterations

# UAT Scenario 6.2: Resume from Interruption
def test_uat_resume_interruption(valid_config, tmp_path):
    valid_config.orchestrator.max_iterations = 5

    # State is loaded as TRAINING
    state = WorkflowState(
        project_name="UAT_Cycle06",
        dataset_path=tmp_path / "data.pckl",
        current_phase=WorkflowPhase.TRAINING,
        generation=0
    )

    with patch("mlip_autopipec.orchestration.manager.StateManager") as MockStateMgr, \
         patch.object(WorkflowManager, 'train', return_value=Path("new.yace")) as mock_train, \
         patch.object(WorkflowManager, 'validate', return_value=True):

        MockStateMgr.return_value.load.return_value = state

        manager = WorkflowManager(valid_config, work_dir=tmp_path)

        # It should execute TRAINING
        manager.step()

        assert mock_train.called
        assert manager.state.current_phase == WorkflowPhase.VALIDATION
