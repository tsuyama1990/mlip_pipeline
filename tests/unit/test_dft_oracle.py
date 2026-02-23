"""Unit tests for DFT Oracle (Mock)."""

import pytest

from pyacemaker.core.config import DFTConfig
from pyacemaker.domain_models.models import StructureMetadata, StructureStatus

# Import from expected location
try:
    from pyacemaker.oracle.dft import DFTOracle
except ImportError:
    DFTOracle = None


@pytest.mark.skipif(DFTOracle is None, reason="DFTOracle not implemented")
def test_dft_oracle_mock_compute() -> None:
    """Test DFT Oracle mock computation."""
    # Setup minimal valid config
    config = DFTConfig(
        code="mock",
        command="mock",
        pseudopotentials={"Fe": "Fe.pbe"},
        parameters={"mock": True}
    )
    oracle = DFTOracle(config, mock=True)

    from ase import Atoms
    # Generate dummy structures
    structures = [
        StructureMetadata(features={"atoms": Atoms("Fe2", positions=[[0,0,0], [2,0,0]])})
        for _ in range(3)
    ]

    # Run computation
    results = list(oracle.compute_batch(structures))

    assert len(results) == 3
    for s in results:
        assert s.status == StructureStatus.CALCULATED
        assert s.energy is not None
        assert s.forces is not None
        assert s.label_source == "dft"

        # Check specific mock behavior (e.g. LJ energy)
        # Should be consistent
        assert s.energy != 0.0  # Unless coincidentally 0
