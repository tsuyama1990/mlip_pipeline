import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from mlip_autopipec.domain_models.config import (
    ACEConfig, Config, OrchestratorConfig, LoggingConfig, PotentialConfig,
    BulkStructureGenConfig, MDConfig, DFTConfig, TrainingConfig, ValidationConfig
)
from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.domain_models.dynamics import LammpsResult
from mlip_autopipec.domain_models.structure import Structure
import numpy as np

@pytest.fixture
def valid_config(tmp_path):
    return Config(
        project_name="UAT_Cycle06",
        logging=LoggingConfig(level="INFO", file_path=tmp_path / "log.txt"),
        orchestrator=OrchestratorConfig(max_iterations=1, uncertainty_threshold=0.1, validation_frequency=1),
        potential=PotentialConfig(
            elements=["Si"],
            cutoff=5.0,
            seed=42,
            pair_style="hybrid/overlay",
            ace_params=ACEConfig(
                npot="FinnisSinclair",
                fs_parameters=[1, 1, 1, 0.5],
                ndensity=2
            )
        ),
        structure_gen=BulkStructureGenConfig(strategy="bulk", element="Si", crystal_structure="diamond", lattice_constant=5.43, supercell=(1,1,1)),
        md=MDConfig(temperature=300, n_steps=100, ensemble="NVT", timestep=0.001),
        dft=DFTConfig(command="pw.x", pseudopotentials={"Si": "Si.upf"}, ecutwfc=40.0, kspacing=0.04),
        training=TrainingConfig(batch_size=10, max_epochs=1, work_dir=tmp_path / "training_work"),
        validation=ValidationConfig(report_path=tmp_path/"report.html")
    )

# UAT Scenario 6.1: The Autonomous Loop
def test_uat_autonomous_loop(valid_config, tmp_path):

    # Mock StateManager and Phases
    with patch("mlip_autopipec.orchestration.orchestrator.StateManager") as MockStateMgr, \
         patch("mlip_autopipec.orchestration.orchestrator.ExplorationPhase") as MockExploration, \
         patch("mlip_autopipec.orchestration.orchestrator.SelectionPhase") as MockSelection, \
         patch("mlip_autopipec.orchestration.orchestrator.CalculationPhase") as MockCalculation, \
         patch("mlip_autopipec.orchestration.orchestrator.TrainingPhase") as MockTraining, \
         patch("mlip_autopipec.orchestration.orchestrator.ValidationPhase") as MockValidation:

        # Setup Mocks
        # Initial State
        state = WorkflowState(
            project_name="UAT_Cycle06",
            dataset_path=tmp_path / "data" / "data.pckl.gzip",
            current_phase=WorkflowPhase.EXPLORATION,
            generation=0
        )
        MockStateMgr.return_value.load.return_value = state

        dummy_struct = Structure(
            symbols=["Si"], positions=np.array([[0,0,0]]), cell=np.eye(3), pbc=(True,True,True)
        )

        # Exploration returns result with high gamma to trigger selection
        MockExploration.return_value.execute.return_value = LammpsResult(
            job_id="test", status=JobStatus.COMPLETED, work_dir=tmp_path, duration_seconds=1,
            log_content="", max_gamma=1.0, # > 0.1 threshold
            final_structure=dummy_struct,
            trajectory_path=tmp_path / "traj.lammpstrj"
        )

        # Selection returns dummy candidates
        MockSelection.return_value.execute.return_value = [MagicMock()]

        # Calculation returns True (success)
        MockCalculation.return_value.execute.return_value = True

        # Training returns path
        MockTraining.return_value.execute.return_value = Path("new.yace")

        # Validation returns True
        MockValidation.return_value.execute.return_value = True

        orchestrator = Orchestrator(valid_config, work_dir=tmp_path)

        # Run loop (simulated by calling step repeatedly)

        # Gen 0: EXPLORATION -> SELECTION
        orchestrator.step()
        assert orchestrator.state.current_phase == WorkflowPhase.SELECTION
        assert MockExploration.return_value.execute.called

        # Gen 0: SELECTION -> CALCULATION
        orchestrator.step()
        assert orchestrator.state.current_phase == WorkflowPhase.CALCULATION
        assert MockSelection.return_value.execute.called

        # Gen 0: CALCULATION -> TRAINING
        orchestrator.step()
        assert orchestrator.state.current_phase == WorkflowPhase.TRAINING
        assert MockCalculation.return_value.execute.called

        # Gen 0: TRAINING -> VALIDATION
        orchestrator.step()
        assert orchestrator.state.current_phase == WorkflowPhase.VALIDATION
        assert MockTraining.return_value.execute.called

        # Gen 0: VALIDATION -> Gen 1 EXPLORATION
        orchestrator.step()
        assert orchestrator.state.generation == 1
        assert orchestrator.state.current_phase == WorkflowPhase.EXPLORATION
        assert MockValidation.return_value.execute.called

        # Check max iterations reached
        assert orchestrator.state.generation >= valid_config.orchestrator.max_iterations

# UAT Scenario 6.2: Resume from Interruption
def test_uat_resume_interruption(valid_config, tmp_path):
    valid_config.orchestrator.max_iterations = 5

    # State is loaded as TRAINING
    state = WorkflowState(
        project_name="UAT_Cycle06",
        dataset_path=tmp_path / "data" / "data.pckl.gzip",
        current_phase=WorkflowPhase.TRAINING,
        generation=0
    )

    with patch("mlip_autopipec.orchestration.orchestrator.StateManager") as MockStateMgr, \
         patch("mlip_autopipec.orchestration.orchestrator.TrainingPhase") as MockTraining, \
         patch("mlip_autopipec.orchestration.orchestrator.ValidationPhase") as MockValidation:

        MockStateMgr.return_value.load.return_value = state
        MockTraining.return_value.execute.return_value = Path("new.yace")
        MockValidation.return_value.execute.return_value = True

        # Dummy exploration mock to satisfy constructor
        with patch("mlip_autopipec.orchestration.orchestrator.ExplorationPhase"), \
             patch("mlip_autopipec.orchestration.orchestrator.SelectionPhase"), \
             patch("mlip_autopipec.orchestration.orchestrator.CalculationPhase"):

            orchestrator = Orchestrator(valid_config, work_dir=tmp_path)

            # It should execute TRAINING
            orchestrator.step()

            assert MockTraining.return_value.execute.called
            assert orchestrator.state.current_phase == WorkflowPhase.VALIDATION
