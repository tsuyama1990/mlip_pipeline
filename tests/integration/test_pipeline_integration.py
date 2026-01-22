from ase import Atoms

from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.surrogate.candidate_manager import CandidateManager


def test_integration_pipeline_real_db(tmp_path):
    """
    Test Pipeline with real DB and mock model but integrated CandidateManager.
    """
    db_path = tmp_path / "integration.db"
    db = DatabaseManager(db_path)
    db.initialize()

    cm = CandidateManager(db)

    # 1. Create candidates
    atoms = Atoms('H2', positions=[[0,0,0], [0,0,0.8]])
    cm.create_candidate(atoms)

    assert db.count(selection="status=pending") == 1

    # 2. Run Pipeline (using mock model for simplicity but exercising DB code)
    from unittest.mock import MagicMock

    import numpy as np

    from mlip_autopipec.config.schemas.surrogate import SurrogateConfig
    from mlip_autopipec.surrogate.pipeline import SurrogatePipeline

    config = SurrogateConfig(model_type="mock", n_samples=1)

    # We still use a mock model to avoid heavy ML deps in this test environment if needed,
    # but the focus is on the DB interaction flow.
    mock_model = MagicMock()
    mock_model.compute_energy_forces.return_value = (np.array([-5.0]), [np.zeros((2,3))])
    mock_model.compute_descriptors.return_value = np.array([[0.0]])

    pipeline = SurrogatePipeline(db, config, model=mock_model)
    pipeline.run()

    # 3. Verify
    assert db.count(selection="status=selected") == 1
    # Check if metadata updated
    entry = db.get_atoms(selection="status=selected")[0]
    assert "mace_energy" in entry.info
