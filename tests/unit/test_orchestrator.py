from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest
from mlip_autopipec.domain_models import (
    BulkStructureGenConfig,
    Config,
    LammpsConfig,
    LoggingConfig,
    MDConfig,
    OrchestratorConfig,
    PotentialConfig,
    JobStatus,
    LammpsResult,
    Structure,
    DFTResult,
    JobResult,
    DFTConfig,
    TrainingConfig
)
from mlip_autopipec.orchestration.orchestrator import Orchestrator
import numpy as np

@pytest.fixture
def mock_config():
    return Config(
        project_name="TestProject",
        logging=LoggingConfig(),
        potential=PotentialConfig(elements=["Si"], cutoff=5.0, pair_style="hybrid/overlay"),
        structure_gen=BulkStructureGenConfig(
            strategy="bulk",
            element="Si",
            crystal_structure="diamond",
            lattice_constant=5.43,
        ),
        md=MDConfig(temperature=300, n_steps=100, ensemble="NVT"),
        lammps=LammpsConfig(),
        orchestrator=OrchestratorConfig(max_iterations=1, uncertainty_threshold=5.0, validation_frequency=1),
        dft=DFTConfig(pseudopotentials={"Si": Path("Si.upf")}, ecutwfc=40.0, kspacing=0.05),
        training=TrainingConfig()
    )

@pytest.fixture
def mock_structure():
    return Structure(
        symbols=["Si"],
        positions=np.array([[0, 0, 0]]),
        cell=np.eye(3),
        pbc=(True, True, True),
    )

def test_orchestrator_one_shot(mock_config, mock_structure):
    """Test standard one-shot execution."""
    with patch("mlip_autopipec.orchestration.orchestrator.StructureGenFactory") as mock_gen_factory, \
         patch("mlip_autopipec.orchestration.orchestrator.LammpsRunner") as mock_runner_cls:

        # Setup mocks
        mock_generator = MagicMock()
        mock_generator.generate.return_value = mock_structure
        mock_gen_factory.get_generator.return_value = mock_generator

        mock_runner = MagicMock()
        mock_runner.run.return_value = LammpsResult(
            job_id="test",
            status=JobStatus.COMPLETED,
            work_dir=".",
            duration_seconds=1.0,
            log_content="ok",
            final_structure=mock_structure,
            trajectory_path="traj.lammpstrj"
        )
        mock_runner_cls.return_value = mock_runner

        # Run
        orchestrator = Orchestrator(mock_config)
        result = orchestrator.run_pipeline()

        assert result.status == JobStatus.COMPLETED
        mock_generator.generate.assert_called_once()
        mock_runner.run.assert_called_once()

def test_orchestrator_loop_with_uncertainty(mock_config, mock_structure):
    """Test loop logic when uncertainty is detected (simulated)."""
    # Enable multiple iterations
    mock_config.orchestrator.max_iterations = 2

    with patch("mlip_autopipec.orchestration.orchestrator.StructureGenFactory") as mock_gen_factory, \
         patch("mlip_autopipec.orchestration.orchestrator.LammpsRunner") as mock_runner_cls:

        mock_generator = MagicMock()
        mock_generator.generate.return_value = mock_structure
        mock_gen_factory.get_generator.return_value = mock_generator

        mock_runner = MagicMock()
        # First run: High uncertainty -> should trigger select/refine (mocked) and continue
        # Second run: Low uncertainty -> break

        # Note: LammpsResult needs valid Path objects usually, but pydantic converts str to Path if allowed
        # Actually in test, we pass args.
        from pathlib import Path

        res1 = LammpsResult(
            job_id="1", status=JobStatus.COMPLETED, work_dir=Path("."), duration_seconds=1.0,
            log_content="ok", final_structure=mock_structure, trajectory_path=Path("t1"), max_gamma=10.0
        )
        res2 = LammpsResult(
            job_id="2", status=JobStatus.COMPLETED, work_dir=Path("."), duration_seconds=1.0,
            log_content="ok", final_structure=mock_structure, trajectory_path=Path("t2"), max_gamma=1.0
        )

        mock_runner.run.side_effect = [res1, res2]
        mock_runner_cls.return_value = mock_runner

        orchestrator = Orchestrator(mock_config)

        # We need to mock select/refine/validate or spy on them
        with patch.object(orchestrator, 'select') as mock_select, \
             patch.object(orchestrator, 'refine') as mock_refine, \
             patch.object(orchestrator, 'validate') as mock_validate:

            result = orchestrator.run_pipeline()

            assert orchestrator.iteration == 2
            assert mock_select.call_count == 1 # Only after first detection
            assert mock_refine.call_count == 1
            # Validate is called after refine if frequency matches
            # Config has frequency 1, so it should be called in iteration 1 (after refine)
            assert mock_validate.call_count == 1

            assert result.job_id == "2"

def test_orchestrator_partial_dft_failure(mock_config, mock_structure):
    """Test that Orchestrator proceeds if some DFTs fail but not all."""
    mock_config.orchestrator.max_iterations = 1

    # Create two candidates
    candidates = [mock_structure.model_copy(), mock_structure.model_copy()]

    with patch("mlip_autopipec.orchestration.orchestrator.QERunner") as mock_qe_runner_cls, \
         patch("mlip_autopipec.orchestration.orchestrator.PacemakerRunner") as mock_pace_runner_cls, \
         patch("mlip_autopipec.orchestration.orchestrator.DatasetManager") as _: # mock_dm_cls removed as unused

        mock_dft_runner = MagicMock()
        # First DFT succeeds, Second fails with generic JobResult (not DFTResult) or DFTResult with dummy
        # If QERunner failure returns JobResult(status=FAILED)
        res_ok = DFTResult(job_id="1", status=JobStatus.COMPLETED, work_dir=".", duration_seconds=1, log_content="ok", energy=-10.0, forces=np.zeros((1,3)), stress=np.zeros((3,3)))
        res_fail = JobResult(job_id="2", status=JobStatus.FAILED, work_dir=".", duration_seconds=1, log_content="error")

        mock_dft_runner.run.side_effect = [res_ok, res_fail]
        mock_qe_runner_cls.return_value = mock_dft_runner

        mock_pace_runner = MagicMock()
        mock_pace_runner.train.return_value = MagicMock(status=JobStatus.COMPLETED, potential_path=Path("new.yace"))
        mock_pace_runner_cls.return_value = mock_pace_runner

        # We also need to mock dataset manager or rely on real one? Real one needs files.
        # Let's mock it on the orchestrator instance

        orchestrator = Orchestrator(mock_config)
        orchestrator.dataset_manager = MagicMock()

        # Mocking dft_dir
        iter_dir = Path("test_iter")
        (iter_dir / "dft_calc").mkdir(parents=True, exist_ok=True)

        # Run Refine directly
        new_pot = orchestrator.refine(candidates, iter_dir)

        # Assertions
        assert new_pot == Path("new.yace")
        assert mock_dft_runner.run.call_count == 2
        # Dataset manager should only be called with 1 result (the successful one)
        orchestrator.dataset_manager.convert.assert_called_once()
        args, _ = orchestrator.dataset_manager.convert.call_args
        assert len(args[0]) == 1 # List of 1 structure
