from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest
import numpy as np

from mlip_autopipec.domain_models import (
    Config,
    LoggingConfig,
    PotentialConfig,
    MDConfig,
    LammpsConfig,
    OrchestratorConfig,
    DFTConfig,
    TrainingConfig,
    Structure,
    DFTResult,
    JobStatus,
    JobResult,
    BulkStructureGenConfig
)
from mlip_autopipec.orchestration.phases import PhaseRefinement, PhaseDetection
from mlip_autopipec.physics.dynamics.lammps import LammpsResult

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

def test_phase_refinement_partial_dft_failure(mock_config, mock_structure):
    """Test that Refinement proceeds if some DFTs fail but not all."""
    candidates = [mock_structure.model_copy(), mock_structure.model_copy()]

    with patch("mlip_autopipec.orchestration.phases.QERunner") as mock_qe_runner_cls, \
         patch("mlip_autopipec.orchestration.phases.PacemakerRunner") as mock_pace_runner_cls, \
         patch("mlip_autopipec.orchestration.phases.DatasetManager") as mock_dm_cls:

        mock_dft_runner = MagicMock()
        res_ok = DFTResult(job_id="1", status=JobStatus.COMPLETED, work_dir=".", duration_seconds=1, log_content="ok", energy=-10.0, forces=np.zeros((1,3)), stress=np.zeros((3,3)))
        res_fail = JobResult(job_id="2", status=JobStatus.FAILED, work_dir=".", duration_seconds=1, log_content="error")

        mock_dft_runner.run.side_effect = [res_ok, res_fail]
        mock_qe_runner_cls.return_value = mock_dft_runner

        mock_pace_runner = MagicMock()
        mock_pace_runner.train.return_value = MagicMock(status=JobStatus.COMPLETED, potential_path=Path("new.yace"))
        mock_pace_runner_cls.return_value = mock_pace_runner

        mock_dm = mock_dm_cls.return_value

        phase = PhaseRefinement(mock_config, data_dir=Path("data"))
        iter_dir = Path("test_iter")

        new_pot = phase.execute(candidates, iter_dir, current_potential_path=None)

        assert new_pot == Path("new.yace")
        assert mock_dft_runner.run.call_count == 2
        mock_dm.convert.assert_called_once()
        args, _ = mock_dm.convert.call_args
        assert len(args[0]) == 1 # Only successful one

def test_phase_refinement_force_masking(mock_config, mock_structure):
    """Test force masking logic in Refinement."""
    s_ghost = mock_structure.model_copy()
    s_ghost.arrays = {"ghost_mask": np.array([True])}
    candidates = [s_ghost]

    with patch("mlip_autopipec.orchestration.phases.QERunner") as mock_qe_runner_cls, \
         patch("mlip_autopipec.orchestration.phases.PacemakerRunner") as mock_pace_runner_cls, \
         patch("mlip_autopipec.orchestration.phases.DatasetManager") as mock_dm_cls:

        mock_dft_runner = MagicMock()
        # Non-zero forces
        res_ok = DFTResult(
            job_id="1", status=JobStatus.COMPLETED, work_dir=".", duration_seconds=1,
            log_content="ok", energy=-10.0,
            forces=np.array([[1.0, 1.0, 1.0]]),
            stress=np.zeros((3,3))
        )
        mock_dft_runner.run.return_value = res_ok
        mock_qe_runner_cls.return_value = mock_dft_runner

        mock_pace_runner = MagicMock()
        mock_pace_runner.train.return_value = MagicMock(status=JobStatus.COMPLETED, potential_path=Path("new.yace"))
        mock_pace_runner_cls.return_value = mock_pace_runner

        mock_dm = mock_dm_cls.return_value

        phase = PhaseRefinement(mock_config, data_dir=Path("data"))
        phase.execute(candidates, Path("test_iter_mask"), None)

        args, _ = mock_dm.convert.call_args
        struct_saved = args[0][0]
        assert np.allclose(struct_saved.properties['forces'], 0.0)

def test_phase_detection(mock_config, mock_structure):
    phase = PhaseDetection(mock_config)

    # High Gamma
    res = LammpsResult(
        job_id="1", status=JobStatus.COMPLETED, work_dir=Path("."), duration_seconds=1,
        log_content="", final_structure=mock_structure, trajectory_path=Path("."),
        max_gamma=10.0
    )
    assert phase.execute(res) is True

    # Low Gamma
    res.max_gamma = 1.0
    assert phase.execute(res) is False

    # None Gamma
    res.max_gamma = None
    assert phase.execute(res) is False
