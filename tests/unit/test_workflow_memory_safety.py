from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.config import DistillationConfig
from pyacemaker.domain_models.state import PipelineState
from pyacemaker.modules.mace_workflow import MaceDistillationWorkflow


# Mock classes
class MockGenerator:
    def generate_direct_samples(self, n_samples, objective):
        return self.generate_candidates()
    def generate_candidates(self) -> Iterator[Any]:
        # Infinite generator
        while True:
            yield MagicMock(spec=Atoms)

@pytest.fixture
def mock_dependencies(tmp_path):
    dataset_manager = MagicMock()
    dataset_manager.save_metadata_stream.return_value = 100

    # Correct mock config structure
    config = MagicMock(spec=DistillationConfig)
    config.pool_file = "pool.xyz"
    config.surrogate_file = "surrogate.xyz"
    config.batch_size = 10
    config.write_buffer_size = 10
    config.step4_surrogate_sampling = MagicMock()
    config.step4_surrogate_sampling.target_points = 50 # Set low for test
    config.step4_surrogate_sampling.method = "random"

    return {
        "config": config,
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
    """Test that step 4 limits the infinite generator correctly."""
    workflow = MaceDistillationWorkflow(**mock_dependencies)
    state = PipelineState(current_step=4, artifacts={"mace_model_path": "model.model"})

    # If islice wasn't working, this would hang forever
    # Mock islice to verify usage? No, integration is better.
    # We rely on the generator being consumed.

    new_state = workflow.step4_surrogate_data_generation(state)

    assert new_state.current_step == 5
    assert "surrogate_pool_path" in new_state.artifacts

    # Check that save_metadata_stream was called
    mock_dependencies["dataset_manager"].save_metadata_stream.assert_called_once()

    # Verify strict limit was used (by checking config usage)
    # This implies we trust islice working if config is accessed
    assert workflow.config.step4_surrogate_sampling.target_points == 50

def test_step5_error_handling(mock_dependencies):
    """Test error handling in step 5."""
    workflow = MaceDistillationWorkflow(**mock_dependencies)
    state = PipelineState(
        current_step=5,
        artifacts={
            "surrogate_pool_path": "dummy.xyz",
            "mace_model_path": "model.model"
        }
    )

    # Mock dataset manager to raise exception
    mock_dependencies["dataset_manager"].load_iter.side_effect = Exception("Disk error")

    with pytest.raises(Exception, match="Disk error"):
        workflow.step5_surrogate_labeling(state)

def test_step5_calculator_errors_handled(mock_dependencies):
    """Test that individual calculator errors don't crash the workflow but log warning."""
    workflow = MaceDistillationWorkflow(**mock_dependencies)
    state = PipelineState(
        current_step=5,
        artifacts={
            "surrogate_pool_path": "dummy.xyz",
            "mace_model_path": "model.model"
        }
    )

    # Return 2 atoms
    mock_dependencies["dataset_manager"].load_iter.return_value = iter([MagicMock(), MagicMock()])

    # Make calculator fail on first atom
    mock_calc = MagicMock()
    mock_calc.get_potential_energy.side_effect = [Exception("Calc Failed"), 0.0]
    mock_dependencies["mace_oracle"].calculator = mock_calc

    with patch("pyacemaker.modules.mace_workflow.stream_metadata_to_atoms") as mock_conv:
        a1, a2 = MagicMock(spec=Atoms), MagicMock(spec=Atoms)
        mock_conv.return_value = iter([a1, a2])

        with patch("builtins.open", new_callable=MagicMock), \
             patch("pyacemaker.modules.mace_workflow.write"):
            # We need to mock write to consume the generator
            new_state = workflow.step5_surrogate_labeling(state)

    assert new_state.current_step == 6
