from unittest.mock import MagicMock

import pytest

from mlip_autopipec.config.schemas.surrogate import SurrogateConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.surrogate.pipeline import SurrogatePipeline


@pytest.fixture
def mock_db():
    return MagicMock(spec=DatabaseManager)

@pytest.fixture
def config():
    return SurrogateConfig(model_type="mock", model_path="model.pt", force_threshold=1.0, n_samples=10)

def test_surrogate_pipeline_empty_input(mock_db, config):
    pipeline = SurrogatePipeline(mock_db, config)
    # Mock no pending entries
    mock_db.get_entries.return_value = []

    pipeline.run()

    # Should exit gracefully
    mock_db.get_entries.assert_called_once()
    # No updates
    mock_db.update_status.assert_not_called()

def test_surrogate_pipeline_model_shape_mismatch(mock_db, config):
    pipeline = SurrogatePipeline(mock_db, config)

    # One pending entry
    mock_db.get_entries.return_value = [(1, "atom_placeholder")]

    # Mock model
    mock_model = MagicMock()
    pipeline.model = mock_model

    # Return 2 energies for 1 atom -> mismatch
    mock_model.compute_energy_forces.return_value = ([1.0, 2.0], [[[0,0,0]], [[0,0,0]]])

    with pytest.raises(RuntimeError, match="mismatches"):
        pipeline.run()
