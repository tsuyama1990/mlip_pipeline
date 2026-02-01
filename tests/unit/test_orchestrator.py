from unittest.mock import patch
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
    with patch("mlip_autopipec.orchestration.orchestrator.PhaseExploration") as mock_exp_cls, \
         patch("mlip_autopipec.orchestration.orchestrator.PhaseDetection") as mock_det_cls, \
         patch("mlip_autopipec.orchestration.orchestrator.PhaseSelection") as mock_sel_cls, \
         patch("mlip_autopipec.orchestration.orchestrator.PhaseRefinement") as mock_ref_cls, \
         patch("mlip_autopipec.orchestration.orchestrator.PhaseValidation") as _:

        # Setup mocks
        mock_exp = mock_exp_cls.return_value
        mock_det = mock_det_cls.return_value

        explore_res = LammpsResult(
            job_id="test",
            status=JobStatus.COMPLETED,
            work_dir=Path("."),
            duration_seconds=1.0,
            log_content="ok",
            final_structure=mock_structure,
            trajectory_path=Path("traj.lammpstrj"),
            max_gamma=1.0
        )
        mock_exp.execute.return_value = explore_res

        # Detection returns False (no uncertainty)
        mock_det.execute.return_value = False

        # Run
        orchestrator = Orchestrator(mock_config)
        result = orchestrator.run_pipeline()

        assert result.status == JobStatus.COMPLETED
        mock_exp.execute.assert_called_once()
        mock_det.execute.assert_called_once()

        # Should not call select/refine
        mock_sel_cls.return_value.execute.assert_not_called()
        mock_ref_cls.return_value.execute.assert_not_called()

def test_orchestrator_loop_with_uncertainty(mock_config, mock_structure):
    """Test loop logic when uncertainty is detected."""
    # Enable multiple iterations
    mock_config.orchestrator.max_iterations = 2

    with patch("mlip_autopipec.orchestration.orchestrator.PhaseExploration") as mock_exp_cls, \
         patch("mlip_autopipec.orchestration.orchestrator.PhaseDetection") as mock_det_cls, \
         patch("mlip_autopipec.orchestration.orchestrator.PhaseSelection") as mock_sel_cls, \
         patch("mlip_autopipec.orchestration.orchestrator.PhaseRefinement") as mock_ref_cls, \
         patch("mlip_autopipec.orchestration.orchestrator.PhaseValidation") as mock_val_cls:

        mock_exp = mock_exp_cls.return_value
        mock_det = mock_det_cls.return_value
        mock_sel = mock_sel_cls.return_value
        mock_ref = mock_ref_cls.return_value
        mock_val = mock_val_cls.return_value

        res1 = LammpsResult(
            job_id="1", status=JobStatus.COMPLETED, work_dir=Path("."), duration_seconds=1.0,
            log_content="ok", final_structure=mock_structure, trajectory_path=Path("t1"), max_gamma=10.0
        )
        res2 = LammpsResult(
            job_id="2", status=JobStatus.COMPLETED, work_dir=Path("."), duration_seconds=1.0,
            log_content="ok", final_structure=mock_structure, trajectory_path=Path("t2"), max_gamma=1.0
        )

        # Iteration 1 returns res1, Iteration 2 returns res2
        mock_exp.execute.side_effect = [res1, res2]

        # Detection: True first, False second
        mock_det.execute.side_effect = [True, False]

        # Selection returns dummy candidates
        mock_sel.execute.return_value = [mock_structure]

        # Refinement returns a new potential path
        mock_ref.execute.return_value = Path("new_pot.yace")

        orchestrator = Orchestrator(mock_config)
        orchestrator.run_pipeline()

        assert orchestrator.iteration == 2
        assert mock_sel.execute.call_count == 1 # Only after first detection
        assert mock_ref.execute.call_count == 1
        assert mock_val.execute.call_count == 1

        # Check that refinement potential was passed to next exploration
        # Call args for explore: (iter_dir, potential_path)
        # First call: initial potential (None or configured)
        # Second call: "new_pot.yace"
        args_list = mock_exp.execute.call_args_list
        assert args_list[1][0][1] == Path("new_pot.yace")
