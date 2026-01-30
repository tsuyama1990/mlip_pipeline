from unittest.mock import MagicMock, patch

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
    Structure
)
from mlip_autopipec.domain_models.training import TrainingConfig, TrainingResult
from mlip_autopipec.domain_models.calculation import DFTConfig, DFTResult
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.orchestration.orchestrator import Orchestrator
import numpy as np
from datetime import datetime
from pathlib import Path

@pytest.fixture
def mock_config():
    return Config(
        project_name="TestProject",
        logging=LoggingConfig(),
        potential=PotentialConfig(elements=["Si"], cutoff=5.0),
        structure_gen=BulkStructureGenConfig(
            strategy="bulk",
            element="Si",
            crystal_structure="diamond",
            lattice_constant=5.43,
        ),
        md=MDConfig(temperature=300, n_steps=100, ensemble="NVT"),
        lammps=LammpsConfig(),
        orchestrator=OrchestratorConfig(max_iterations=1, uncertainty_threshold=5.0),
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
            work_dir=Path("."),
            duration_seconds=1.0,
            log_content="ok",
            final_structure=mock_structure,
            trajectory_path=Path("traj.lammpstrj")
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

        # We need to mock select/refine or spy on them
        with patch.object(orchestrator, 'select') as mock_select, \
             patch.object(orchestrator, 'refine') as mock_refine:

            # mock select return value
            mock_select.return_value = [mock_structure]

            result = orchestrator.run_pipeline()

            assert orchestrator.iteration == 2
            assert mock_select.call_count == 1 # Only after first detection
            assert mock_refine.call_count == 1
            assert result.job_id == "2"

def test_orchestrator_refine(mock_config, mock_structure):
    """Test the refine phase (Label -> Train)."""
    mock_config.dft = DFTConfig(
        command="pw.x", pseudopotentials={"Si": Path("Si.upf")}, ecutwfc=30, kspacing=0.04
    )
    mock_config.training = TrainingConfig(
        max_epochs=1, batch_size=1, ladder_step=[1, 1], kappa=0.5
    )

    orchestrator = Orchestrator(mock_config)

    candidates = [mock_structure]

    with patch("mlip_autopipec.orchestration.orchestrator.QERunner") as MockQE, \
         patch("mlip_autopipec.orchestration.orchestrator.DatasetManager") as MockDM, \
         patch("mlip_autopipec.orchestration.orchestrator.PacemakerRunner") as MockPace:

        # Mock QE
        qe_instance = MockQE.return_value
        qe_instance.run.return_value = DFTResult(
            job_id="dft", status=JobStatus.COMPLETED, work_dir=Path("."), duration_seconds=1.0,
            log_content="ok", energy=-10.0, forces=np.zeros((1,3))
        )

        # Mock DM
        dm_instance = MockDM.return_value
        dm_instance.convert.return_value = Path("dataset.pckl.gzip")

        # Mock Pace
        pace_instance = MockPace.return_value
        pace_instance.select_active_set.return_value = Path("active.pckl.gzip")
        pace_instance.train.return_value = TrainingResult(
            job_id="train", status=JobStatus.COMPLETED, work_dir=Path("."), duration_seconds=1.0,
            log_content="ok",
            potential=Potential(path=Path("pot.yace"), format="ace", elements=["Si"], creation_date=datetime.now(), metadata={}),
            validation_metrics={}
        )

        result = orchestrator.refine(candidates)

        assert isinstance(result, TrainingResult)
        assert result.potential.path == Path("pot.yace")

        qe_instance.run.assert_called()
        dm_instance.convert.assert_called()
        pace_instance.train.assert_called()
