from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.domain_models.models import Potential, PotentialType, StructureMetadata
from pyacemaker.domain_models.state import PipelineState
from pyacemaker.modules.mace_workflow import MaceDistillationWorkflow


@pytest.fixture
def mock_config() -> MagicMock:
    config = MagicMock()
    config.batch_size = 10
    config.write_buffer_size = 10
    # Step 7 config
    config.step7_pacemaker_finetune.enable = True
    config.step7_pacemaker_finetune.weight_dft = 10.0
    # Step 1 config
    config.step1_direct_sampling.target_points = 5
    config.step1_direct_sampling.objective = "test"
    # Step 4 config
    config.step4_surrogate_sampling.target_points = 5
    return config

@pytest.fixture
def mock_dataset_manager() -> MagicMock:
    dm = MagicMock()
    # Mock save_metadata_stream to return count
    dm.save_metadata_stream.return_value = 5
    return dm

@pytest.fixture
def mock_active_learner() -> MagicMock:
    al = MagicMock()
    al.run_loop.return_value = (Path("model.mace"), Path("dft_data.xyz"))
    return al

@pytest.fixture
def mock_mace_oracle() -> MagicMock:
    oracle = MagicMock()
    # Mock compute_batch to return iterator
    def compute_side_effect(iterator: Any) -> Any:
        yield from iterator
    oracle.compute_batch.side_effect = compute_side_effect
    return oracle

@pytest.fixture
def mock_pacemaker_trainer() -> MagicMock:
    trainer = MagicMock()
    trainer.train.return_value = Potential(
        path=Path("final.yace"),
        type=PotentialType.PACE,
        version="1.0.0",
        metrics={},
        parameters={}
    )
    return trainer

@pytest.fixture
def workflow(mock_config: MagicMock, mock_dataset_manager: MagicMock, mock_active_learner: MagicMock, mock_mace_oracle: MagicMock, mock_pacemaker_trainer: MagicMock, tmp_path: Path) -> MaceDistillationWorkflow:
    return MaceDistillationWorkflow(
        config=mock_config,
        dataset_manager=mock_dataset_manager,
        active_learner=mock_active_learner,
        structure_generator=MagicMock(),
        oracle=MagicMock(),
        mace_oracle=mock_mace_oracle,
        pacemaker_trainer=mock_pacemaker_trainer,
        mace_trainer=MagicMock(),
        work_dir=tmp_path
    )

def test_step2_active_learning_saves_dft_path(workflow: MaceDistillationWorkflow) -> None:
    state = PipelineState(current_step=2)
    state.artifacts["pool_path"] = "pool.xyz"

    new_state = workflow.step2_active_learning_loop(state)

    assert new_state.artifacts["dft_dataset_path"] == "dft_data.xyz"
    assert new_state.artifacts["mace_model_path"] == "model.mace"

def test_step5_surrogate_labeling_calls_compute_batch(workflow: MaceDistillationWorkflow, mock_mace_oracle: MagicMock, mock_dataset_manager: MagicMock) -> None:
    state = PipelineState(current_step=5)
    state.artifacts["surrogate_pool_path"] = "surrogate.xyz"
    state.artifacts["mace_model_path"] = "model.mace"

    # Mock dataset load
    s1 = StructureMetadata()
    mock_dataset_manager.load_iter.return_value = iter([s1])

    # Mock stream_metadata_to_atoms
    with patch("pyacemaker.modules.mace_workflow.stream_metadata_to_atoms") as mock_conv:
        mock_conv.return_value = iter([MagicMock()]) # Atoms iterator

        # Mock _write_labeled_stream to avoid file I/O
        workflow._write_labeled_stream = MagicMock(return_value=1)

        workflow.step5_surrogate_labeling(state)

        # Verify compute_batch was called with the iterator from load_iter
        assert mock_mace_oracle.compute_batch.called
        args, _ = mock_mace_oracle.compute_batch.call_args
        # The argument should be the iterator returned by load_iter
        assert args[0] is mock_dataset_manager.load_iter.return_value

        assert workflow._write_labeled_stream.called

def test_step5_handles_error(workflow: MaceDistillationWorkflow, mock_mace_oracle: MagicMock, mock_dataset_manager: MagicMock) -> None:
    state = PipelineState(current_step=5)
    state.artifacts["surrogate_pool_path"] = "surrogate.xyz"
    state.artifacts["mace_model_path"] = "model.mace"

    mock_dataset_manager.load_iter.return_value = iter([StructureMetadata()])

    # Mock compute_batch to raise exception
    mock_mace_oracle.compute_batch.side_effect = RuntimeError("Oracle Failed")

    with pytest.raises(RuntimeError, match="Oracle Failed"):
        workflow.step5_surrogate_labeling(state)

def test_step7_delta_learning_calls_trainer(workflow: MaceDistillationWorkflow, mock_pacemaker_trainer: MagicMock, mock_dataset_manager: MagicMock) -> None:
    state = PipelineState(current_step=7)
    state.artifacts["pacemaker_potential_path"] = "base.yace"
    state.artifacts["dft_dataset_path"] = "dft_data.xyz"

    # Mock dataset load (returns atoms)
    atom = MagicMock()
    mock_dataset_manager.load_iter.return_value = iter([atom])

    # Use patch for atoms_to_metadata since it's imported in the module
    with patch("pyacemaker.modules.mace_workflow.atoms_to_metadata") as mock_conv:
        mock_conv.return_value = StructureMetadata()

        new_state = workflow.step7_delta_learning(state)

        assert mock_pacemaker_trainer.train.called
        args, kwargs = mock_pacemaker_trainer.train.call_args
        assert kwargs["weight_dft"] == 10.0
        assert kwargs["initial_potential"].path == Path("base.yace")
        assert new_state.artifacts["final_potential"] == "final.yace"
