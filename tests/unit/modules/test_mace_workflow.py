"""Tests for MaceDistillationWorkflow."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from ase import Atoms

from pyacemaker.core.config import DistillationConfig
from pyacemaker.domain_models.state import PipelineState
from pyacemaker.modules.mace_workflow import MaceDistillationWorkflow


@pytest.fixture
def mock_workflow(tmp_path: Path) -> MaceDistillationWorkflow:
    """Fixture for MaceDistillationWorkflow."""
    config = DistillationConfig(
        enable_mace_distillation=True,
        pool_file="pool.xyz",
        surrogate_file="surrogate.xyz",
        surrogate_dataset_file="dataset.xyz",
    )
    return MaceDistillationWorkflow(
        config=config,
        dataset_manager=MagicMock(),
        active_learner=MagicMock(),
        structure_generator=MagicMock(),
        oracle=MagicMock(),
        mace_oracle=MagicMock(),
        pacemaker_trainer=MagicMock(),
        mace_trainer=MagicMock(),
        work_dir=tmp_path,
    )


def test_step5_surrogate_labeling(mock_workflow: MaceDistillationWorkflow) -> None:
    """Test Step 5: Surrogate Labeling with batched processing."""
    state = PipelineState(current_step=5)
    state.artifacts["surrogate_pool_path"] = str(mock_workflow.work_dir / "surrogate.xyz")
    state.artifacts["mace_model_path"] = str(mock_workflow.work_dir / "model.model")

    # Mock dataset manager to return an iterator of atoms
    atoms = [Atoms("Si2"), Atoms("Si2"), Atoms("Si2")]
    mock_workflow.dataset_manager.load_iter.return_value = iter(atoms)

    # Mock batch processing
    # We need to mock _process_batch or the oracle calculator
    mock_workflow.mace_oracle.calculator = MagicMock()

    # We patch stream_metadata_to_atoms because load_iter returns metadata usually?
    # Actually load_iter returns Atoms if format is ase-supported, or metadata if custom format.
    # The workflow calls stream_metadata_to_atoms.
    # Let's mock stream_metadata_to_atoms
    with patch("pyacemaker.modules.mace_workflow.stream_metadata_to_atoms") as mock_stream:
        mock_stream.return_value = iter(atoms)

        # We need to mock validate_safe_path
        with patch("pyacemaker.modules.mace_workflow.validate_safe_path", return_value=mock_workflow.work_dir / "safe.xyz"):
             # Execute
             new_state = mock_workflow.step5_surrogate_labeling(state)

    # Verify
    assert new_state.current_step == 6
    assert "labeled_surrogate_path" in new_state.artifacts
    # Verify calculator was attached and used
    assert atoms[0].calc == mock_workflow.mace_oracle.calculator

    # We can verify batch processing logic by checking calls to calculator if we mocked it deeper,
    # but checking state transition and artifact creation is key for integration.


def test_step7_delta_learning_disabled(mock_workflow: MaceDistillationWorkflow) -> None:
    """Test Step 7: Delta Learning (Disabled)."""
    mock_workflow.config.step7_pacemaker_finetune.enable = False
    state = PipelineState(current_step=7)

    new_state = mock_workflow.step7_delta_learning(state)

    assert new_state.current_step == 8


def test_step7_delta_learning_enabled(mock_workflow: MaceDistillationWorkflow) -> None:
    """Test Step 7: Delta Learning (Enabled)."""
    mock_workflow.config.step7_pacemaker_finetune.enable = True
    state = PipelineState(current_step=7)
    state.artifacts["pacemaker_potential_path"] = "pot.yace"

    new_state = mock_workflow.step7_delta_learning(state)

    assert new_state.current_step == 8
    assert new_state.artifacts["final_potential"] == "pot.yace"
