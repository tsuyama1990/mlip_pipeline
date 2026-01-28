from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config.models import (
    DFTConfig,
    MLIPConfig,
    RuntimeConfig,
    TargetSystem,
    WorkflowConfig,
)
from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.config.schemas.training import TrainingConfig
from mlip_autopipec.domain_models.structure_enums import CandidateStatus
from mlip_autopipec.orchestration.database import DatabaseManager
from mlip_autopipec.orchestration.workflow import WorkflowManager

# UAT Scenario 06-01: Full Active Learning Cycle (Simulated)


@pytest.fixture
def uat_config(tmp_path):
    (tmp_path / "Fe.UPF").touch()

    train_conf = TrainingConfig(
        training_data_path=str(tmp_path / "train.xyz"),
        test_data_path=str(tmp_path / "test.xyz"),
        cutoff=5.0,
        b_basis_size=100,
        kappa=0.5,
        kappa_f=10.0,
        batch_size=32,
    )

    inf_conf = InferenceConfig(lammps_executable="/bin/lmp", temperature=100.0, steps=100)

    return MLIPConfig(
        target_system=TargetSystem(name="UAT System", elements=["Fe"], composition={"Fe": 1.0}),
        dft=DFTConfig(
            pseudopotential_dir=tmp_path, ecutwfc=30, kspacing=0.05, command="pw.x"
        ),
        workflow=WorkflowConfig(max_generations=2),
        runtime=RuntimeConfig(work_dir=tmp_path / "_work", database_path="mlip.db"),
        training_config=train_conf,
        inference_config=inf_conf,
    )


def test_uat_full_cycle_simulation(uat_config, tmp_path):
    work_dir = uat_config.runtime.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "current.yace").touch()
    (tmp_path / "new.yace").touch()

    with DatabaseManager(work_dir / uat_config.runtime.database_path) as db:
        atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[2.5, 2.5, 2.5], pbc=True)
        db.add_structure(atoms, {"status": CandidateStatus.TRAINING.value})
        assert db.count() == 1
        assert db.count(selection=f"status={CandidateStatus.TRAINING.value}") == 1

    with (
        patch("mlip_autopipec.orchestration.phases.dft.QERunner"),
        patch("mlip_autopipec.orchestration.phases.exploration.LammpsRunner") as MockLammpsRunner,
        patch("mlip_autopipec.orchestration.phases.exploration.StructureBuilder") as MockBuilder,
        patch(
            "mlip_autopipec.orchestration.phases.training.PacemakerWrapper"
        ) as MockPacemakerWrapper,
        patch(
            "mlip_autopipec.orchestration.phases.selection.PacemakerWrapper"
        ) as MockSelectionPacemaker,
        patch("mlip_autopipec.inference.processing.EmbeddingExtractor") as MockExtractor,
        patch("mlip_autopipec.inference.processing.read") as mock_proc_read,
        patch("mlip_autopipec.surrogate.candidate_manager.read") as mock_cm_read,
        patch("mlip_autopipec.orchestration.phases.training.DatasetBuilder"),
        patch("mlip_autopipec.orchestration.workflow.TaskQueue") as MockTaskQueue,
    ):
        # Configure StructureBuilder
        mock_builder = MockBuilder.return_value
        # Return list of 2 Atoms to trigger LammpsRunner twice in Cycle 1
        mock_builder.build.return_value = [
            Atoms("Fe", positions=[[0, 0, 0]], cell=[2.5, 2.5, 2.5], pbc=True),
            Atoms("Fe", positions=[[0, 0, 0]], cell=[2.5, 2.5, 2.5], pbc=True)
        ]

        # Configure LammpsRunner
        mock_lammps = MockLammpsRunner.return_value
        result_halt = MagicMock()
        dump_file = tmp_path / "dump.1"
        dump_file.touch()
        result_halt.uncertain_structures = [dump_file]

        result_converged = MagicMock()
        result_converged.uncertain_structures = []

        # We have 2 calls expected (one per atom). Both halt or one halts.
        mock_lammps.run.side_effect = [result_halt, result_halt]

        # Configure ase.read
        atoms_with_gamma = Atoms("Fe", positions=[[0, 0, 0]], cell=[2.5, 2.5, 2.5], pbc=True)
        atoms_with_gamma.arrays["c_gamma"] = np.array([20.0])
        mock_proc_read.return_value = atoms_with_gamma
        mock_cm_read.return_value = atoms_with_gamma

        # Configure EmbeddingExtractor
        mock_extractor = MockExtractor.return_value
        mock_extractor.extract.return_value = Atoms(
            "Fe", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True
        )

        # Configure DFT Runner (Wait, Phase uses TaskQueue!)
        dft_res = MagicMock(
            energy=-10.0,
            forces=[[0, 0, 0]],
            stress=[[0] * 3] * 3,
            converged=True,
            succeeded=True,
            error_message=None,
            uid="mock_uid",
            wall_time=1.0,
            parameters={}
        )

        # Configure TaskQueue
        mock_task_queue = MockTaskQueue.return_value

        # wait_for_completion takes futures and returns list of results
        # We need to return list of dft_res matching input
        def wait_side_effect(futures):
            # We assume futures length matches input length
            return [dft_res] * len(futures)

        mock_task_queue.wait_for_completion.side_effect = wait_side_effect
        # Also need submit_dft_batch to return dummy futures
        mock_task_queue.submit_dft_batch.side_effect = lambda func, atoms_list: [MagicMock()] * len(
            atoms_list
        )

        # Configure Pacemaker (Training)
        mock_pacemaker = MockPacemakerWrapper.return_value
        mock_pacemaker.train.return_value.success = True
        mock_pacemaker.train.return_value.potential_path = tmp_path / "new.yace"

        # Configure Pacemaker (Selection)
        mock_sel_pacemaker = MockSelectionPacemaker.return_value
        mock_sel_pacemaker.select_active_set.return_value = [0]

        # Run
        manager = WorkflowManager(uat_config, work_dir=work_dir, workflow_config=uat_config.workflow_config)
        # Seed the state with an initial potential for the first inference cycle
        manager.state.latest_potential_path = work_dir / "current.yace"
        manager.run()

        # Assertions
        assert manager.state.cycle_index == 2

        assert mock_lammps.run.call_count == 2

        # Verify TaskQueue submit was called instead of QE Runner directly
        assert mock_task_queue.submit_dft_batch.called
