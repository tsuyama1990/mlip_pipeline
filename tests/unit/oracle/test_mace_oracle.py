"""Tests for MaceSurrogateOracle."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.domain_models.models import StructureMetadata, StructureStatus
from pyacemaker.oracle.mace_oracle import MaceSurrogateOracle


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    # We avoid spec=PYACEMAKERConfig because it causes issues with nested fields in mocks sometimes
    config = MagicMock()
    config.oracle = MagicMock()
    config.oracle.mace = MagicMock()
    config.oracle.mock = False
    config.oracle.mace.model_path = "medium"
    return config


@pytest.fixture
def mock_mace_manager():
    """Mock MaceManager."""
    with patch("pyacemaker.oracle.mace_oracle.MaceManager") as mock:
        yield mock


def test_mace_oracle_init(mock_config, mock_mace_manager):
    """Test initialization."""
    oracle = MaceSurrogateOracle(mock_config)
    assert oracle.mace_manager is not None
    mock_mace_manager.assert_called_once()


def test_predict_batch(mock_config, mock_mace_manager):
    """Test predict_batch."""
    oracle = MaceSurrogateOracle(mock_config)
    manager_instance = mock_mace_manager.return_value

    # Prepare input
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True)
    meta = StructureMetadata(features={"atoms": atoms})
    structures = [meta]

    # Mock result
    result_atoms = atoms.copy()
    # Attach a mock calculator so get_potential_energy works
    mock_calc = MagicMock()
    mock_calc.get_potential_energy.return_value = -5.0
    mock_calc.get_forces.return_value = np.array([[0, 0, 0.1], [0, 0, -0.1]])
    mock_calc.get_stress.return_value = np.array([0.0] * 6)
    result_atoms.calc = mock_calc

    manager_instance.compute.return_value = result_atoms

    # Run
    processed = oracle.predict_batch(structures)

    assert len(processed) == 1
    s = processed[0]
    assert s.energy == -5.0
    assert s.forces == [[0, 0, 0.1], [0, 0, -0.1]]
    assert s.stress == [0.0] * 6
    assert s.status == StructureStatus.CALCULATED
    assert s.label_source == "mace"


def test_predict_batch_mock_mode(mock_config):
    """Test predict_batch in mock mode."""
    mock_config.oracle.mock = True
    oracle = MaceSurrogateOracle(mock_config)
    assert oracle.mace_manager is None

    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    meta = StructureMetadata(features={"atoms": atoms})
    structures = [meta]

    processed = oracle.predict_batch(structures)
    assert processed[0].energy == -10.0
    assert processed[0].status == StructureStatus.CALCULATED
    assert processed[0].label_source == "mace"


def test_compute_uncertainty(mock_config, mock_mace_manager):
    """Test compute_uncertainty."""
    oracle = MaceSurrogateOracle(mock_config)
    manager_instance = mock_mace_manager.return_value

    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    meta = StructureMetadata(features={"atoms": atoms})
    structures = [meta]

    manager_instance.compute_uncertainty.return_value = [0.1]

    processed = list(oracle.compute_uncertainty(structures))
    assert len(processed) == 1
    assert processed[0].uncertainty_state.gamma_mean == 0.1
    assert processed[0].uncertainty_state.gamma_max == 0.1
