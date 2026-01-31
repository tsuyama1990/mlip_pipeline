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
    Structure,
    TrainingConfig,
    DFTConfig,
)
from mlip_autopipec.orchestration.orchestrator import Orchestrator
import numpy as np
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
        orchestrator=OrchestratorConfig(
            max_iterations=1, uncertainty_threshold=5.0, max_active_set_size=5
        ),
        dft=DFTConfig(
            command="pw.x",
            pseudopotentials={"Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF"},
            ecutwfc=30.0,
            kspacing=0.1,
        ),
        training=TrainingConfig(batch_size=10, max_epochs=10),
    )


@pytest.fixture
def mock_structure():
    return Structure(
        symbols=["Si"],
        positions=np.array([[0, 0, 0]]),
        cell=np.eye(3),
        pbc=(True, True, True),
    )


def test_orchestrator_select_streaming(mock_config, mock_structure):
    """Test that select uses streaming (iread)."""
    orchestrator = Orchestrator(mock_config)

    # Create a result pointing to a fake file
    res = LammpsResult(
        job_id="test",
        status=JobStatus.COMPLETED,
        work_dir=Path("."),
        duration_seconds=1.0,
        log_content="ok",
        final_structure=mock_structure,
        trajectory_path=Path("fake_traj.lammpstrj"),
    )

    # Mock ase.io.iread
    # We simulate a trajectory with some high gamma frames
    atoms1 = mock_structure.to_ase()
    atoms1.arrays["c_pace_gamma"] = np.array([1.0])  # low

    atoms2 = mock_structure.to_ase()
    atoms2.arrays["c_pace_gamma"] = np.array([6.0])  # high (threshold 5.0)

    atoms3 = mock_structure.to_ase()
    atoms3.arrays["c_pace_gamma"] = np.array([2.0])  # low

    with (
        patch(
            "ase.io.iread", return_value=iter([atoms1, atoms2, atoms3])
        ) as mock_iread,
        patch.object(Path, "exists", return_value=True),
    ):
        candidates = orchestrator.select(res)

        mock_iread.assert_called_once()
        assert len(candidates) == 1
        # Check if we got the correct one (can't check equality easily since Structure conversion creates new obj,
        # but we can check if it was derived from atoms2)
        # Assuming only atoms2 selected.


def test_orchestrator_refine_generator(mock_config, mock_structure):
    """Test that refine uses generator for DFT results."""
    orchestrator = Orchestrator(mock_config)

    candidates = [mock_structure, mock_structure]
    iter_dir = Path("test_iter")

    with (
        patch("mlip_autopipec.orchestration.orchestrator.QERunner") as mock_qe_cls,
        patch(
            "mlip_autopipec.orchestration.orchestrator.PacemakerRunner"
        ) as mock_pace_cls,
        patch.object(orchestrator.dataset_manager, "convert") as mock_convert,
        patch.object(Path, "mkdir"),
    ):
        mock_qe_runner = MagicMock()
        mock_qe_cls.return_value = mock_qe_runner

        # Mock run results
        from mlip_autopipec.domain_models.calculation import DFTResult

        dft_res = DFTResult(
            job_id="dft",
            status=JobStatus.COMPLETED,
            work_dir=Path("."),
            duration_seconds=1.0,
            log_content="ok",
            energy=-10.0,
            forces=np.zeros((1, 3)),
            stress=np.zeros(6),
        )
        mock_qe_runner.run.return_value = dft_res

        mock_pace_runner = MagicMock()
        mock_pace_cls.return_value = mock_pace_runner
        mock_pace_runner.train.return_value = MagicMock(
            status=JobStatus.COMPLETED, potential_path=Path("new.yace")
        )

        # Call refine
        orchestrator.refine(candidates, iter_dir)

        # Check that convert was called with a generator
        mock_convert.assert_called_once()
        args, _ = mock_convert.call_args
        structure_generator = args[0]

        # Verify it is an iterator/generator
        from typing import Iterator

        assert isinstance(structure_generator, Iterator)

        # Now consume the generator to verify it calls QERunner
        # Note: Since refine passes the generator to convert, and convert consumes it,
        # normally orchestrator.refine would have already triggered the consumption IF convert was not mocked.
        # But here convert IS mocked. So the generator was passed but NOT consumed yet.
        # So QERunner.run should NOT have been called yet!

        mock_qe_runner.run.assert_not_called()

        # Consume generator
        results = list(structure_generator)
        assert len(results) == 2
        assert mock_qe_runner.run.call_count == 2
