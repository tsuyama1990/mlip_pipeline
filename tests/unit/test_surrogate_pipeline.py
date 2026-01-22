from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.surrogate import SurrogateConfig
from mlip_autopipec.surrogate.pipeline import SurrogatePipeline


@pytest.fixture
def mock_db():
    db = MagicMock()
    # Mock get_entries to return some pending atoms
    atoms1 = Atoms('H2', positions=[[0,0,0], [0,0,0.8]])
    atoms2 = Atoms('O2', positions=[[0,0,0], [0,0,1.2]])
    # db.get_entries returns list of (id, atoms)
    db.get_entries.return_value = [
        (1, atoms1),
        (2, atoms2)
    ]
    return db

@pytest.fixture
def mock_model():
    model = MagicMock()
    # Mock energy/forces
    # Atom 1: Good (force=0). Atom 2: High force (force=100)
    # Energy: -10, -5
    # Forces: list of (N,3) arrays
    f1 = np.zeros((2,3))
    f2 = np.ones((2,3)) * 100.0

    model.compute_energy_forces.return_value = (
        np.array([-10.0, -5.0]),
        [f1, f2]
    )
    # Mock descriptors
    model.compute_descriptors.return_value = np.array([[0,0], [1,1]])
    return model

def test_pipeline_run(mock_db, mock_model):
    config = SurrogateConfig(
        model_type="mock",
        force_threshold=50.0,
        n_samples=1
    )
    # In real code, pipeline might create model internally.
    # We will assume we can inject it or patch it.
    # Here we inject it via constructor dependency injection (common pattern).

    pipeline = SurrogatePipeline(db_manager=mock_db, config=config, model=mock_model)

    pipeline.run()

    # Check Model calls
    mock_model.load_model.assert_called()
    mock_model.compute_energy_forces.assert_called()

    # Check DB updates
    # ID 2 should be REJECTED (force 100 > 50)
    # We check if update_status was called with ID 2 and 'rejected'
    mock_db.update_status.assert_any_call(2, "rejected")

    # ID 1 should be SELECTED (since we want n_samples=1 and it's valid)
    mock_db.update_status.assert_any_call(1, "selected")

def test_pipeline_fps_selection(mock_db, mock_model):
    """Test FPS selection when candidates > n_samples"""
    # 3 valid atoms
    config = SurrogateConfig(
        model_type="mock",
        force_threshold=50.0,
        n_samples=1
    )

    atoms1 = Atoms('H2', positions=[[0,0,0], [0,0,0.8]]) # Valid
    atoms2 = Atoms('H2', positions=[[0,0,0], [0,0,0.9]]) # Valid
    atoms3 = Atoms('H2', positions=[[0,0,0], [0,0,1.0]]) # Valid

    mock_db.get_entries.return_value = [
        (1, atoms1),
        (2, atoms2),
        (3, atoms3)
    ]

    mock_model.compute_energy_forces.return_value = (
        np.zeros(3),
        [np.zeros((2,3))]*3
    )
    mock_model.compute_descriptors.return_value = np.array([[0,0], [1,1], [2,2]])

    pipeline = SurrogatePipeline(mock_db, config, model=mock_model)
    pipeline.run()

    # n_samples=1. So 1 selected, 2 held.
    # We check if update_status was called with 'held'
    # mock_db.update_status calls: (id, "selected") once, and (id, "held") twice.

    # Verify calls
    # We can check specific calls or counts
    calls = mock_db.update_status.call_args_list
    statuses = [c[0][1] for c in calls]
    assert statuses.count("selected") == 1
    assert statuses.count("held") == 2

def test_pipeline_no_pending(mock_db):
    mock_db.get_entries.return_value = []
    pipeline = SurrogatePipeline(mock_db, SurrogateConfig())
    pipeline.run()
    mock_db.update_status.assert_not_called()
