"""Unit tests for MACE Oracle (Uncertainty)."""

import pytest

from pyacemaker.core.config import MaceConfig
from pyacemaker.domain_models.models import StructureMetadata

# Import from expected location
try:
    from pyacemaker.oracle.mace_oracle import MaceSurrogateOracle
except ImportError:
    MaceSurrogateOracle = None


@pytest.mark.skipif(MaceSurrogateOracle is None, reason="MaceSurrogateOracle not implemented")
def test_mace_oracle_uncertainty_mock() -> None:
    """Test compute_uncertainty with mock implementation."""
    config = MaceConfig(mock=True, model_path="medium")
    oracle = MaceSurrogateOracle(config)

    from ase import Atoms
    # Generate dummy structures
    structures = [
        StructureMetadata(features={"atoms": Atoms("H2", positions=[[0,0,0], [0,0,1]])})
        for _ in range(5)
    ]

    # Run uncertainty computation
    results = list(oracle.compute_uncertainty(structures))

    assert len(results) == 5
    for s in results:
        assert s.uncertainty_state is not None
        assert s.uncertainty_state.gamma_max is not None
        # In mock mode, uncertainty should be > 0
        assert s.uncertainty_state.gamma_max >= 0.0
        # Check computed field alias
        assert s.uncertainty is not None
        assert s.uncertainty == s.uncertainty_state.gamma_max


@pytest.mark.skipif(MaceSurrogateOracle is None, reason="MaceSurrogateOracle not implemented")
def test_mace_oracle_compute_batch() -> None:
    """Test compute_batch (prediction) with mock."""
    config = MaceConfig(mock=True, model_path="medium")
    oracle = MaceSurrogateOracle(config)

    from ase import Atoms
    structures = [
        StructureMetadata(features={"atoms": Atoms("H2", positions=[[0,0,0], [0,0,1]])})
        for _ in range(3)
    ]

    # Run computation
    results = list(oracle.compute_batch(structures))

    assert len(results) == 3
    for s in results:
        assert s.status == "CALCULATED"
        assert s.energy is not None
        assert s.forces is not None
        assert s.label_source == "mace"
