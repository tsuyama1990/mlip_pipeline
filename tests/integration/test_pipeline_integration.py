
import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.surrogate import SurrogateConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.surrogate.pipeline import SurrogatePipeline


@pytest.fixture
def mock_db(tmp_path):
    """
    Creates a temporary database for integration testing.
    This replaces any 'real' DB file with an isolated temp file.
    """
    db_path = tmp_path / "test_pipeline.db"
    return DatabaseManager(db_path)

def test_integration_pipeline_real_db(mock_db):
    """
    Integration test using a real (but temporary) SQLite database file.
    Verifies interaction between DB and Surrogate Pipeline.
    """
    # 1. Setup DB with pending structures
    with mock_db:
        mock_db.add_structure(Atoms('H2', positions=[[0,0,0], [0,0,0.74]]), metadata={"status": "pending"})
        mock_db.add_structure(Atoms('O2', positions=[[0,0,0], [0,0,1.2]]), metadata={"status": "pending"})
        assert mock_db.count(status="pending") == 2

    # 2. Configure Surrogate Pipeline (using mock model for simplicity in this integration scope)
    config = SurrogateConfig(
        model_type="mock",
        model_path="dummy.pt",
        force_threshold=100.0,
        n_samples=1
    )

    # Mocking the MaceWrapper internally used by pipeline to avoid heavy model loading
    # We patch at the class level or inject a mock model if the constructor allows.
    # SurrogatePipeline allows injecting `model`.

    from unittest.mock import MagicMock
    mock_model_interface = MagicMock()
    # Mock return: energy per atom, forces per atom
    # H2: 2 atoms. O2: 2 atoms.
    mock_model_interface.compute_energy_forces.return_value = (
        [-1.0, -2.0], # Energies
        [[[0,0,0], [0,0,0]], [[0,0,0], [0,0,0]]] # Zero forces
    )
    # Mock descriptors
    mock_model_interface.compute_descriptors.return_value = [[1.0], [2.0]]

    pipeline = SurrogatePipeline(mock_db, config, model=mock_model_interface)

    # 3. Run Pipeline
    pipeline.run()

    # 4. Verify DB state
    with mock_db:
        # 1 selected (n_samples=1), 1 held
        assert mock_db.count(status="selected") == 1
        assert mock_db.count(status="held") == 1
        assert mock_db.count(status="pending") == 0
