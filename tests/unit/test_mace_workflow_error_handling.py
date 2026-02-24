"""Tests for MACE Workflow Error Handling."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.domain_models.models import (
    Potential,
    PotentialType,
    StructureMetadata,
    UncertaintyState,
)
from pyacemaker.modules.mace_workflow import MaceDistillationWorkflow
from pyacemaker.oracle.dataset import DatasetManager
from pyacemaker.oracle.mace_oracle import MaceSurrogateOracle


@pytest.fixture
def mock_config(tmp_path: Path) -> PYACEMAKERConfig:
    config = MagicMock(spec=PYACEMAKERConfig)
    config.distillation = MagicMock()
    config.distillation.enable_mace_distillation = True
    config.distillation.step2_active_learning.cycles = 2
    config.distillation.step2_active_learning.uncertainty_threshold = 0.5
    config.distillation.step2_active_learning.n_select = 5
    config.oracle = MagicMock()
    config.oracle.mace.model_path = "medium"
    # Ensure nested mocks exist
    config.project = MagicMock()
    config.project.root_dir = tmp_path
    return config


def test_mace_workflow_oracle_failure_recovery(mock_config: PYACEMAKERConfig, tmp_path: Path) -> None:
    """Test that workflow continues if one AL iteration fails due to Oracle error."""
    mock_sg = MagicMock()
    # Mock Step 1 to return a pool
    dataset_manager = DatasetManager()
    pool_path = tmp_path / "pool.pckl.gzip"
    # Create dummy pool using generator to simulate streaming and avoid memory loading
    from ase import Atoms
    atoms_iter = (Atoms("H") for _ in range(10))
    dataset_manager.save_iter(atoms_iter, pool_path)

    # Mock generator return value also as iterator
    mock_sg.generate_direct_samples.return_value = (
        StructureMetadata(features={"atoms": Atoms("H")}, id=i) for i in range(10)
    )

    mock_dft_oracle = MagicMock()
    mock_mace_oracle = MagicMock(spec=MaceSurrogateOracle)

    # Mock uncertainty to fail on first call, succeed on second
    def uncertainty_side_effect(structures):
        # We need to yield metadata, but fail on first call
        if mock_mace_oracle.compute_uncertainty.call_count == 1:
            err_msg = "Transient Failure"
            raise RuntimeError(err_msg)
        # Return dummy structures
        for s in structures:
            s.uncertainty_state = UncertaintyState(gamma_max=0.9, gamma_mean=0.9)
            yield s

    mock_mace_oracle.compute_uncertainty.side_effect = uncertainty_side_effect
    mock_mace_oracle.compute_batch.side_effect = lambda s: s

    # Mock Trainer
    mock_trainer = MagicMock()
    mock_mace_trainer = MagicMock()
    mock_mace_trainer.train.return_value = Potential(path=Path("mace.model"), type=PotentialType.MACE, version="1.0", metrics={}, parameters={})

    # Mock Dynamics
    mock_dyn = MagicMock()
    mock_dyn.run_exploration.return_value = []

    workflow = MaceDistillationWorkflow(
        config=mock_config,
        dataset_manager=dataset_manager,
        dataset_path=tmp_path / "dataset.pckl.gzip",
        oracle=mock_dft_oracle,
        mace_oracle=mock_mace_oracle,
        trainer=mock_trainer,
        mace_trainer=mock_mace_trainer,
        dynamics_engine=mock_dyn,
        structure_generator=mock_sg,
        validation_path=tmp_path / "val",
        training_path=tmp_path / "train",
        active_learner=MagicMock(),
    )

    # Patch Step 1 to return our pool path
    with patch.object(workflow, "step1_direct_sampling", return_value=pool_path):
        result = workflow.run()

    # Should succeed overall (despite AL failure)
    assert result.status == "success"
    # Verify we tried
    assert mock_mace_oracle.compute_uncertainty.called
