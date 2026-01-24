from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.config.models import DFTConfig, SystemConfig, TargetSystem, WorkflowConfig
from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.config.schemas.training import TrainingConfig
from mlip_autopipec.orchestration.workflow import WorkflowManager

# UAT Scenario 06-01: Full Active Learning Cycle (Simulated)


@pytest.fixture
def uat_config(tmp_path):
    (tmp_path / "Fe.UPF").touch()

    # Create valid TrainingConfig
    train_conf = TrainingConfig(
        training_data_path=str(tmp_path / "train.xyz"),
        test_data_path=str(tmp_path / "test.xyz"),
        cutoff=5.0,
        b_basis_size=100,
        kappa=0.5,
        kappa_f=10.0,
        batch_size=32,
    )

    # Create valid InferenceConfig
    inf_conf = InferenceConfig(lammps_executable="/bin/lmp", temperature=100.0, steps=100)

    return SystemConfig(
        target_system=TargetSystem(name="UAT System", elements=["Fe"], composition={"Fe": 1.0}),
        dft_config=DFTConfig(
            pseudopotential_dir=tmp_path, ecutwfc=30, kspacing=0.05, command="pw.x"
        ),
        workflow_config=WorkflowConfig(max_generations=2),
        working_dir=tmp_path / "_work",
        db_path=tmp_path / "mlip.db",
        training_config=train_conf,
        inference_config=inf_conf,
    )


def test_uat_full_cycle_simulation(uat_config, tmp_path):
    # 1. Setup mocks for runners
    # Note: We patch where the classes are IMPORTED/USED, which is now in sub-phases
    with (
        patch("mlip_autopipec.orchestration.phases.dft.QERunner") as MockQERunner,
        patch("mlip_autopipec.orchestration.phases.inference.LammpsRunner") as MockLammpsRunner,
        patch(
            "mlip_autopipec.orchestration.phases.training.PacemakerWrapper"
        ) as MockPacemakerWrapper,
        patch(
            "mlip_autopipec.orchestration.phases.selection.PacemakerWrapper"
        ) as MockSelectionPacemaker,
            patch("mlip_autopipec.inference.processing.EmbeddingExtractor") as MockExtractor,
        patch("mlip_autopipec.orchestration.phases.training.DatasetBuilder") as MockDatasetBuilder,
        patch("mlip_autopipec.orchestration.workflow.TaskQueue") as MockTaskQueue,
        patch("mlip_autopipec.orchestration.workflow.get_dask_client") as mock_dask,
        patch("mlip_autopipec.inference.processing.read") as mock_read,
    ):
        # Configure Read
        from ase import Atoms
        mock_read.return_value = Atoms("Fe")

        # Configure LammpsRunner to trigger Active Learning ONCE
        mock_lammps = MockLammpsRunner.return_value
        # First call: Returns result with uncertain_structures (Halt)
        # Second call: Returns result with NO uncertain_structures (Converged)

        result_halt = MagicMock()
        result_halt.uncertain_structures = [Path("dump.1")]

        result_converged = MagicMock()
        result_converged.uncertain_structures = []

        mock_lammps.run.side_effect = [result_halt, result_converged]

        # Configure EmbeddingExtractor to return atoms
        mock_extractor = MockExtractor.return_value
        from ase import Atoms

        mock_extractor.extract.return_value = Atoms("Fe")

        # Configure DFT Runner
        mock_qe = MockQERunner.return_value
        mock_qe.run.return_value = {"energy": -10.0, "forces": [[0, 0, 0]], "stress": [0] * 6}

        # Configure TaskQueue to return DFT results
        mock_task_queue = MockTaskQueue.return_value
        from mlip_autopipec.data_models.dft_models import DFTResult
        result_dft = DFTResult(
            succeeded=True, energy=-10.0, forces=[[0.0, 0.0, 0.0]], stress=[[0.0]*3]*3,
            uid="123", wall_time=1.0, parameters={}
        )
        def dft_side_effect(atoms, config):
            return [result_dft] * len(atoms)
        mock_task_queue.submit_dft_batch.side_effect = dft_side_effect

        # Configure Pacemaker (Training)
        mock_pacemaker = MockPacemakerWrapper.return_value
        mock_pacemaker.train.return_value.success = True
        mock_pacemaker.train.return_value.potential_path = tmp_path / "new.yace"
        (tmp_path / "new.yace").touch()

        # Configure Pacemaker (Selection)
        mock_sel_pacemaker = MockSelectionPacemaker.return_value
        mock_sel_pacemaker.select_active_set.return_value = [0]

        # 2. Initialize WorkflowManager
        manager = WorkflowManager(uat_config, workflow_config=uat_config.workflow_config)

        # Ensure work directory and initial potential exist
        manager.work_dir.mkdir(parents=True, exist_ok=True)
        (manager.work_dir / "current.yace").touch()

        # Seed DB with a training structure for Inference
        from mlip_autopipec.orchestration.database import DatabaseManager
        from ase import Atoms
        with DatabaseManager(uat_config.db_path) as db:
            db.add_structure(Atoms("Fe"), metadata={"status": "training"})

        # 3. Run
        manager.run()

        # 4. Verify Flow

        # Check State Transitions
        # It should have completed Cycle 0 and Cycle 1?
        # Inference(0) -> Halt -> Selection -> Calculation -> Training(0) -> Cycle=1
        # Inference(1) -> No Halt -> Converged -> Cycle=2 (Max) -> Stop

        assert manager.state.cycle_index == 2

        # Verify Lammps called twice
        assert mock_lammps.run.call_count == 2

        # Verify DFT called
        # We had 1 uncertain structure -> 1 DFT calculation
        assert mock_task_queue.submit_dft_batch.call_count >= 1

        # Verify Training called
        assert mock_pacemaker.train.call_count >= 1
