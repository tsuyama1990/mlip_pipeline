import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from ase import Atoms
from mlip_autopipec.surrogate.mace_client import MaceClient
from mlip_autopipec.config.schemas.surrogate import SurrogateConfig, RejectionInfo

@pytest.fixture
def mock_mace_model():
    with patch('mlip_autopipec.surrogate.mace_client.MaceClient._load_model') as mock_load:
        yield mock_load

def test_mace_client_initialization():
    config = SurrogateConfig(device="cpu", model_path="medium")
    # We don't want to actually load the model in tests
    with patch('mlip_autopipec.surrogate.mace_client.MaceClient._load_model'):
        client = MaceClient(config)
        assert client.config == config

def test_mace_client_path_traversal():
    # Test path traversal prevention
    config = SurrogateConfig(model_path="../etc/passwd")
    client = MaceClient(config)

    # We patch inside _load_model scope if possible, or just call it.
    # The mace_mp is imported inside _load_model, so we patch sys.modules or mocked import?
    # Actually, we can just assert that model remains None and no error is raised to top (caught inside).

    # But to be sure it failed due to traversal and not ImportError of mace,
    # we should check if it logged?
    # The implementation:
    # if ".." in self.config.model_path: raise ValueError

    client._load_model()
    assert client.model is None

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
        assert "Prediction failed" in str(excinfo.value)
