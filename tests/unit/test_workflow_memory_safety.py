from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.dataset import DatasetManager
from pyacemaker.domain_models.state import PipelineState
from pyacemaker.modules.mace_workflow import MaceDistillationWorkflow


# Mock classes
class MockGenerator:
    def generate_candidates(self) -> Iterator[Any]:
        # Infinite generator
        while True:
            yield MagicMock(spec=Atoms)

@pytest.fixture
def mock_dependencies(tmp_path):
    dataset_manager = MagicMock(spec=DatasetManager)
    dataset_manager.save_metadata_stream.return_value = 100

    return {
        "config": MagicMock(),
        "dataset_manager": dataset_manager,
        "active_learner": MagicMock(),
        "structure_generator": MockGenerator(),
        "oracle": MagicMock(),
        "mace_oracle": MagicMock(),
        "pacemaker_trainer": MagicMock(),
        "mace_trainer": MagicMock(),
        "work_dir": tmp_path
    }

def test_step4_surrogate_generation_memory_safety(mock_dependencies):
    """Test that step 4 limits the infinite generator."""
    workflow = MaceDistillationWorkflow(**mock_dependencies)
    state = PipelineState(current_step=4, total_steps=8, artifacts={})

    # If islice wasn't working, this would hang forever
    new_state = workflow.step4_surrogate_data_generation(state)

    assert new_state.current_step == 5
    assert "surrogate_pool_path" in new_state.artifacts
    # Check that save_metadata_stream was called (it consumes the iterator)
    mock_dependencies["dataset_manager"].save_metadata_stream.assert_called_once()

def test_step5_streaming_processing(mock_dependencies):
    """Test that step 5 processes data in streams."""
    workflow = MaceDistillationWorkflow(**mock_dependencies)
    state = PipelineState(
        current_step=5,
        total_steps=8,
        artifacts={
            "surrogate_pool_path": "dummy.xyz",
            "mace_model_path": "model.model"
        }
    )

    # Mock dataset manager to return an iterator of metadata
    mock_dependencies["dataset_manager"].load_iter.return_value = iter([MagicMock() for _ in range(5)])

    # Mock atoms conversion
    with patch("pyacemaker.modules.mace_workflow.stream_metadata_to_atoms") as mock_conv:
        mock_conv.return_value = iter([MagicMock(spec=Atoms) for _ in range(5)])

        with patch("builtins.open", new_callable=MagicMock) as mock_open:
             new_state = workflow.step5_surrogate_labeling(state)

    assert new_state.current_step == 6
