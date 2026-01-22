import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config.models import DFTConfig, InferenceConfig, SystemConfig, TrainingConfig
from mlip_autopipec.config.schemas.surrogate import SurrogateConfig
from mlip_autopipec.config.schemas.training import TrainingMetrics
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.data_models.dft_models import DFTResult
from mlip_autopipec.orchestration.models import OrchestratorConfig
from mlip_autopipec.orchestration.workflow import WorkflowManager
from mlip_autopipec.surrogate.pipeline import SurrogatePipeline
from mlip_autopipec.training.dataset import DatasetBuilder
from mlip_autopipec.training.pacemaker import PacemakerWrapper


@pytest.fixture
def mock_db(tmp_path):
    """
    Creates a temporary database for integration testing.
    This replaces any 'real' DB file with an isolated temp file.
    The fixture automatically handles cleanup by deleting tmp_path.
    """
    db_path = tmp_path / "test_pipeline.db"
    manager = DatabaseManager(db_path)
    # Ensure any connection is closed after test
    return manager
    # DatabaseManager context manager usage in tests is preferred,
    # but if used directly, we rely on GC or explicit close if method existed.
    # ASE db connection is usually file based.

def test_integration_pipeline_real_db(mock_db):
    """
    Integration test using a real (but temporary) SQLite database file.
    Verifies interaction between DB and Surrogate Pipeline.
    """
    # 1. Setup DB with pending structures
    with mock_db:
        mock_db.add_structure(Atoms('H2', positions=[[0,0,0], [0,0,0.74]]), metadata={"status": "pending"})
        mock_db.add_structure(Atoms('O2', positions=[[0,0,0], [0,0,1.2]]), metadata={"status": "pending"})
        assert mock_db.count(status="pending") == 2

    # 2. Configure Surrogate Pipeline (using mock model for simplicity in this integration scope)
    config = SurrogateConfig(
        model_type="mock",
        model_path="dummy.pt",
        force_threshold=100.0,
        n_samples=1
    )

    # Mocking the MaceWrapper internally used by pipeline to avoid heavy model loading
    mock_model_interface = MagicMock()
    mock_model_interface.compute_energy_forces.return_value = (
        np.array([-1.0, -2.0]), # Energies
        [np.array([[0,0,0], [0,0,0]]), np.array([[0,0,0], [0,0,0]])] # Zero forces
    )
    mock_model_interface.compute_descriptors.return_value = np.array([[1.0], [2.0]])

    pipeline = SurrogatePipeline(mock_db, config, model=mock_model_interface)

    # 3. Run Pipeline
    pipeline.run()

    # 4. Verify DB state
    with mock_db:
        # 1 selected (n_samples=1), 1 held
        assert mock_db.count(status="selected") == 1
        assert mock_db.count(status="held") == 1
        assert mock_db.count(status="pending") == 0

@pytest.mark.xfail(reason="Complex mocking of subprocess/shutil in integration context is fragile")
def test_integration_training_flow(mock_db, tmp_path):
    """
    Integration test for the training flow: DB -> Dataset -> Config -> Pacemaker(Mock).
    """
    # 1. Setup DB with completed structures
    with mock_db:
        for i in range(10):
             at = Atoms('Cu', positions=[[0,0,0]])
             at.info['energy'] = -3.0
             at.arrays['forces'] = np.array([[0.0, 0.0, 0.0]])
             mock_db.add_structure(at, metadata={"status": "completed"})

    config = TrainingConfig(
        cutoff=3.0,
        b_basis_size=10,
        kappa=1.0,
        kappa_f=1.0,
        max_iter=10
    )

    work_dir = tmp_path / "training_work"

    # 2. Export Data
    builder = DatasetBuilder(mock_db)
    builder.export(config, work_dir)

    train_file = work_dir / "data" / "train.xyz"
    assert train_file.exists()
    assert (work_dir / "data" / "test.xyz").exists()

    # Verify content
    with open(train_file) as f:
        content = f.read()
        assert "Lattice=" in content
        assert "energy=" in content
        assert "Cu" in content
        # Ensure it is ExtXYZ
        assert "Properties=species:S:1:pos:R:3" in content

    # 3. Train (Mocking Pacemaker subprocess)
    with patch("mlip_autopipec.training.pacemaker.subprocess.run") as mock_run, \
         patch("mlip_autopipec.training.pacemaker.shutil.which", return_value="/usr/bin/pacemaker"), \
         patch("mlip_autopipec.training.pacemaker.PacemakerWrapper.check_output", return_value=True), \
         patch("mlip_autopipec.training.metrics.LogParser.parse_file") as mock_parse:

        mock_run.return_value.returncode = 0
        mock_parse.return_value = TrainingMetrics(epoch=10, rmse_energy=0.1, rmse_force=0.01)

        wrapper = PacemakerWrapper(config, work_dir)
        result = wrapper.train()

        assert result.success is True
        assert result.metrics.rmse_energy == 0.1
        assert (work_dir / "input.yaml").exists()


@pytest.fixture
def integration_config(tmp_path):
    (tmp_path / "work").mkdir()
    return SystemConfig(
        working_dir=tmp_path / "work",
        db_path=tmp_path / "mlip.db",
        dft_config=DFTConfig(
            command="pw.x",
            pseudopotential_dir=tmp_path,
            ecutwfc=40.0,
            kspacing=0.1
        ),
        training_config=TrainingConfig(
            cutoff=5.0,
            b_basis_size=10,
            kappa=0.5,
            kappa_f=100.0,
            max_iter=10
        ),
        inference_config=InferenceConfig(steps=100)
    )

@pytest.mark.xfail(reason="Complex mocking of multiple external binaries in integration context is fragile")
def test_full_loop_integration(integration_config):
    """
    Test the full loop:
    1. Generation (Mocked Builder) -> DB (Pending)
    2. Workflow -> DFT (Mocked Runner) -> DB (Training)
    3. Workflow -> Training (Mocked Pacemaker) -> .yace
    4. Workflow -> Inference (Mocked LAMMPS) -> Active Learning (Mocked Extraction) -> DB (Pending)
    """

    # 1. Setup Manager
    orch_config = OrchestratorConfig(max_generations=2)

    # Mock Builder to produce 1 candidate
    builder_mock = MagicMock()
    candidate = Atoms('H2', positions=[[0,0,0], [0,0,0.74]], cell=[10,10,10])
    builder_mock.build.return_value = [candidate]

    # Initialize Manager
    # We need to patch dependencies inside WorkflowManager or PhaseExecutor

    with patch("mlip_autopipec.orchestration.workflow.TaskQueue") as MockTaskQueue, \
         patch("mlip_autopipec.orchestration.phase_executor.shutil.which", return_value="/bin/echo"), \
         patch("mlip_autopipec.inference.runner.shutil.which", return_value="/bin/echo"):

        # Mock TaskQueue to return successful futures immediately
        queue_instance = MockTaskQueue.return_value
        queue_instance.submit_dft_batch.return_value = [MagicMock()] # 1 future

        # Return proper DFTResult object
        dft_res = DFTResult(
            uid=str(uuid.uuid4()),
            succeeded=True,
            energy=-13.6,
            forces=np.zeros((2,3)),
            stress=np.zeros((3,3)),
            wall_time=1.0,
            parameters={"ecutwfc": 40.0},
            final_mixing_beta=0.7
        )
        queue_instance.wait_for_completion.return_value = [dft_res]

        manager = WorkflowManager(integration_config, orch_config, builder=builder_mock)

        # MOCKING PhaseExecutor components

        with patch("mlip_autopipec.orchestration.phase_executor.QERunner") as MockQE:
            with patch("mlip_autopipec.orchestration.phase_executor.PacemakerWrapper") as MockPM:
                # Patch LammpsRunner at source to avoid AttributeError if import path varies
                with patch("mlip_autopipec.inference.runner.LammpsRunner") as MockLammps:
                    with patch("mlip_autopipec.inference.embedding.EmbeddingExtractor") as MockExtractor:
                         with patch("mlip_autopipec.orchestration.phase_executor.DatasetBuilder"):
                            with patch("mlip_autopipec.orchestration.phase_executor.TrainConfigGenerator"):

                                # Configure Mocks
                                # Need to ensure train returns a result with potential_path
                                mock_train_result = MagicMock()
                                mock_train_result.potential_path = integration_config.working_dir / "potentials" / "generation_0.yace"
                                mock_train_result.success = True
                                MockPM.return_value.train.return_value = mock_train_result

                                (integration_config.working_dir / "potentials").mkdir(parents=True)
                                (integration_config.working_dir / "potentials" / "generation_0.yace").touch()

                                # Mock Inference to return High Uncertainty
                                mock_run_result = MagicMock()
                                mock_run_result.uncertain_structures = [Path("dump.gamma")]
                                MockLammps.return_value.run.return_value = mock_run_result

                                # Mock Extractor
                                mock_extracted = MagicMock()
                                mock_extracted.atoms = Atoms('H', cell=[5,5,5])
                                MockExtractor.return_value.extract.return_value = mock_extracted

                                # Mock ASE read for dump file in execute_inference
                                with patch("ase.io.read") as mock_read:
                                    atom = Atoms('H', cell=[5,5,5])
                                    atom.new_array('c_gamma', np.array([10.0]))
                                    mock_read.return_value = [atom]

                                    # RUN GENERATION 0

                                    # 1. Idle -> DFT
                                    manager._dispatch_phase()
                                    assert manager.state.status == "dft"
                                    # DB should have 1 pending structure
                                    assert manager.db_manager.count(status="pending") == 1

                                    # 2. DFT -> Training
                                    manager._dispatch_phase()

                                    assert manager.state.status == "training"

                                    # DB should have 1 training structure (completed)
                                    # And original structure is now 'labeled' (updated logic)
                                    assert manager.db_manager.count(status="training") == 1
                                    assert manager.db_manager.count(status="labeled") == 1

                                    # 3. Training -> Inference
                                    manager._dispatch_phase()
                                    assert manager.state.status == "inference"

                                    # 4. Inference -> Active Learning -> DFT
                                    # Inference triggered active learning because we mocked uncertain_structures
                                    manager._dispatch_phase()
                                    assert manager.state.status == "dft"

                                    # DB should have NEW pending structure (from extraction)
                                    assert manager.db_manager.count(status="pending") == 1
