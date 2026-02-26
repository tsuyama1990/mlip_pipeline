from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

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
def mock_dataset_manager(tmp_path: Path) -> Any:
    # Use real DatasetManager with a temp file
    from pyacemaker.oracle.dataset import DatasetManager
    return DatasetManager()

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
    state.artifacts["pool_path"] = Path("pool.xyz")

    new_state = workflow.step2_active_learning_loop(state)

    assert new_state.artifacts["dft_dataset_path"] == Path("dft_data.xyz")
    assert new_state.artifacts["mace_model_path"] == Path("model.mace")

def test_step5_surrogate_labeling_calls_compute_batch(workflow: MaceDistillationWorkflow, mock_mace_oracle: MagicMock, tmp_path: Path) -> None:
    state = PipelineState(current_step=5)

    # Create a real surrogate pool file
    pool_path = tmp_path / "surrogate.xyz"
    from ase import Atoms

    from pyacemaker.oracle.dataset import DatasetManager

    dm = DatasetManager()
    atoms_list = [Atoms("H2"), Atoms("H2")]
    dm.save(atoms_list, pool_path)

    state.artifacts["surrogate_pool_path"] = pool_path
    state.artifacts["mace_model_path"] = Path("model.mace")

    # Mock _write_labeled_stream to avoid output file I/O complexity
    workflow._write_labeled_stream = MagicMock(return_value=2)

    # Mock mace_oracle.compute_batch to return metadata iterator
    def mock_compute(iterator: Any) -> Any:
        # yield from iterator but item is StructureMetadata (after fix)
        yield from iterator
    mock_mace_oracle.compute_batch.side_effect = mock_compute

    # Call step5
    workflow.step5_surrogate_labeling(state)

    assert mock_mace_oracle.compute_batch.called
    assert workflow._write_labeled_stream.called

def test_step5_handles_error(workflow: MaceDistillationWorkflow, mock_mace_oracle: MagicMock, tmp_path: Path) -> None:
    state = PipelineState(current_step=5)
    pool_path = tmp_path / "surrogate_err.xyz"

    # Create dummy file
    from ase import Atoms

    from pyacemaker.oracle.dataset import DatasetManager
    DatasetManager().save([Atoms("H")], pool_path)

    state.artifacts["surrogate_pool_path"] = pool_path
    state.artifacts["mace_model_path"] = Path("model.mace")

    # Mock compute_batch to raise exception
    mock_mace_oracle.compute_batch.side_effect = RuntimeError("Oracle Failed")

    # Use check to ensure error message context is logged/raised
    # Since workflow re-raises, we expect RuntimeError
    with pytest.raises(RuntimeError, match="Oracle Failed"):
        workflow.step5_surrogate_labeling(state)

def test_step5_chunked_streaming(workflow: MaceDistillationWorkflow, mock_mace_oracle: MagicMock, tmp_path: Path) -> None:
    """Verify that step5 processes large files in chunks without OOM."""
    state = PipelineState(current_step=5)
    pool_path = tmp_path / "surrogate_large.xyz"

    # Create a larger file (e.g. 100 atoms)
    # We want to ensure that compute_batch receives an iterator, not a list.
    from ase import Atoms

    from pyacemaker.oracle.dataset import DatasetManager

    n_atoms = 100
    atoms_list = [Atoms("H") for _ in range(n_atoms)]
    DatasetManager().save(atoms_list, pool_path)

    state.artifacts["surrogate_pool_path"] = pool_path
    state.artifacts["mace_model_path"] = Path("model.mace")

    # Mock _write_labeled_stream
    workflow._write_labeled_stream = MagicMock(return_value=n_atoms)

    # Mock compute_batch to check if input is an iterator
    def mock_compute_check_iter(iterator: Any) -> Any:
        # Check if it's an iterator (not a list)
        assert iter(iterator) is iterator
        # Consume it
        count = 0
        for _ in iterator:
            count += 1
            yield StructureMetadata()
        assert count == n_atoms

    mock_mace_oracle.compute_batch.side_effect = mock_compute_check_iter

    workflow.step5_surrogate_labeling(state)
    assert mock_mace_oracle.compute_batch.called

def test_step7_delta_learning_calls_trainer(workflow: MaceDistillationWorkflow, mock_pacemaker_trainer: MagicMock, mock_dataset_manager: MagicMock, tmp_path: Path) -> None:
    state = PipelineState(current_step=7)
    state.artifacts["pacemaker_potential_path"] = Path("base.yace")

    # Write a real file for DFT dataset
    from ase import Atoms

    from pyacemaker.oracle.dataset import DatasetManager
    dft_path = tmp_path / "dft_data.xyz"
    DatasetManager().save([Atoms("H")], dft_path)
    state.artifacts["dft_dataset_path"] = dft_path

    # Use real logic for atoms_to_metadata by not mocking it, but ensuring the file content is valid.
    # We already wrote "H" atom to file.

    # We need to verify that what gets passed to trainer.train is an iterator of StructureMetadata
    # derived from that file.

    # Mock trainer.train to consume the dataset iterator and verify contents
    def train_side_effect(dataset, initial_potential, weight_dft):
        # Consume dataset to verify it works
        data_list = list(dataset)
        assert len(data_list) == 1
        assert isinstance(data_list[0], StructureMetadata)
        # Verify weight_dft
        assert weight_dft == 10.0
        return Potential(
            path=Path("final.yace"),
            type=PotentialType.PACE,
            version="1.0.0",
            metrics={},
            parameters={}
        )
    mock_pacemaker_trainer.train.side_effect = train_side_effect

    new_state = workflow.step7_delta_learning(state)

    assert mock_pacemaker_trainer.train.called
    assert new_state.artifacts["final_potential"] == Path("final.yace")
