from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.surrogate import RejectionInfo, SurrogateConfig
from mlip_autopipec.surrogate.mace_client import MaceClient


@pytest.fixture
def mock_mace_model():
    with patch('mlip_autopipec.surrogate.mace_client.MaceClient._load_model') as mock_load:
        yield mock_load

def test_mace_client_initialization():
    """Test that MaceClient initializes with config."""
    config = SurrogateConfig(device="cpu", model_path="medium")
    # We don't want to actually load the model in tests
    with patch('mlip_autopipec.surrogate.mace_client.MaceClient._load_model'):
        client = MaceClient(config)
        assert client.config == config
        assert client.model is None # Initially None

def test_mace_client_load_model_success():
    """Test successful model loading (mocked)."""
    config = SurrogateConfig(device="cpu")
    client = MaceClient(config)

    with patch('mace.calculators.mace_mp') as mock_mace:
        mock_mace.return_value = "MockedModel"
        client._load_model()
        assert client.model == "MockedModel"

def test_mace_client_path_traversal():
    # Test path traversal prevention
    config = SurrogateConfig(model_path="../etc/passwd")
    client = MaceClient(config)

    with pytest.raises(ValueError, match="Path traversal"):
        client._load_model()
    assert client.model is None

def test_predict_forces_explicit():
    """Explicitly test predict_forces method."""
    config = SurrogateConfig(device="cpu")
    atoms_list = [Atoms('H'), Atoms('H')]
    expected_forces = [np.array([[0.0, 0.0, 0.1]]), np.array([[0.0, 0.0, 0.2]])]

    client = MaceClient(config)
    # Mock model
    client.model = MagicMock()
    # When atoms.get_forces() is called, ASE delegates to client.model.get_forces(atoms) usually
    # But here we are swapping calculator.
    # The simplest way to mock this is to mock atoms.get_forces, but atoms is created inside test.
    # Actually, we can just assume ASE works and mock what the model returns.

    # We need to ensure that when we attach client.model to atoms, atoms.get_forces() returns what we want.
    # atoms.get_forces() calls self.calc.get_forces(self) -> self.calc.calculate(...) -> results['forces']

    # Let's mock get_forces on the model directly.
    client.model.get_forces.side_effect = expected_forces

    # We also need to mock _load_model so it doesn't overwrite our mock model
    with patch.object(client, '_load_model'):
        forces = client.predict_forces(atoms_list)

        assert len(forces) == 2
        np.testing.assert_array_equal(forces[0], expected_forces[0])
        np.testing.assert_array_equal(forces[1], expected_forces[1])

def test_filter_unphysical():
    config = SurrogateConfig(force_threshold=10.0, device="cpu")

    atoms_list = [Atoms('H'), Atoms('H'), Atoms('H')]

    mock_forces = [
        np.array([[100.0, 0.0, 0.0]]), # Norm = 100
        np.array([[1.0, 0.0, 0.0]]),   # Norm = 1
        np.array([[5.0, 0.0, 0.0]]),   # Norm = 5
    ]

    with patch('mlip_autopipec.surrogate.mace_client.MaceClient.predict_forces', side_effect=[mock_forces]):
        with patch('mlip_autopipec.surrogate.mace_client.MaceClient._load_model'):
            client = MaceClient(config)
            good_atoms, rejected = client.filter_unphysical(atoms_list)

            assert len(good_atoms) == 2
            assert len(rejected) == 1
            assert isinstance(rejected[0], RejectionInfo)
            assert rejected[0].index == 0
            assert rejected[0].max_force == 100.0

def test_mace_predict_failure():
    config = SurrogateConfig(device="cpu")
    atoms_list = [Atoms('H')]

    with patch('mlip_autopipec.surrogate.mace_client.MaceClient.predict_forces', side_effect=RuntimeError("MACE failed")):
        client = MaceClient(config)
        with pytest.raises(RuntimeError) as excinfo:
            client.filter_unphysical(atoms_list)
        # The exception message wraps the original one.
        # "Pre-screening prediction failed: MACE failed"
        assert "Prediction failed" in str(excinfo.value) or "MACE failed" in str(excinfo.value)
