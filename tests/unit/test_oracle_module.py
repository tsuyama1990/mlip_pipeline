"""Tests for Oracle module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.config import (
    DFTConfig,
    OracleConfig,
    ProjectConfig,
    PYACEMAKERConfig,
)
from pyacemaker.core.exceptions import PYACEMAKERError
from pyacemaker.domain_models.models import StructureMetadata, StructureStatus
from pyacemaker.modules.oracle import DFTOracle, MockOracle


@pytest.fixture
def config() -> PYACEMAKERConfig:
    """Return a valid configuration for DFTOracle."""
    return PYACEMAKERConfig(
        version="0.1.0",
        project=ProjectConfig(name="Test", root_dir=Path()),
        oracle=OracleConfig(
            dft=DFTConfig(
                code="qe", pseudopotentials={"Fe": "Fe.pbe.UPF"}, parameters={}
            ),
            mock=False,
        ),
    )


def test_dft_oracle_compute_batch(config: PYACEMAKERConfig) -> None:
    """Test DFTOracle compute_batch logic."""
    oracle = DFTOracle(config)

    # Create structures with atoms attached
    atoms1 = Atoms("H")
    s1 = StructureMetadata(tags=["test"], features={"atoms": atoms1})

    # Create structure without atoms (should skip)
    s2 = StructureMetadata(tags=["test"])

    structures = [s1, s2]

    # Mock DFTManager.compute_batch
    # It receives list of atoms. Should receive [atoms1].
    # Returns [result_atoms].
    # Use MagicMock for result atoms to avoid method assignment issues
    result_atoms = MagicMock(spec=Atoms)
    result_atoms.get_potential_energy.return_value = -13.6
    # ASE methods return numpy arrays, which have tolist()
    result_atoms.get_forces.return_value = np.array([[0.0, 0.0, 0.0]])
    result_atoms.get_stress.return_value = np.array([0.0] * 6)

    with patch(
        "pyacemaker.oracle.manager.DFTManager.compute_batch", return_value=[result_atoms]
    ) as mock_compute:
        results = oracle.compute_batch(structures)

        assert len(results) == 2

        # s1 should be calculated
        assert results[0].status == StructureStatus.CALCULATED
        assert results[0].features["energy"] == -13.6
        assert results[0].features["atoms"] == result_atoms

        # s2 should remain NEW (skipped)
        assert results[1].status == StructureStatus.NEW

        mock_compute.assert_called_once()
        args = mock_compute.call_args[0][0]
        assert len(args) == 1
        assert args[0] == atoms1


def test_mock_oracle_simulation_failure(config: PYACEMAKERConfig) -> None:
    """Test MockOracle simulating a failure."""
    # Configure simulate_failure
    config.oracle.dft.parameters["simulate_failure"] = True

    oracle = MockOracle(config)

    with pytest.raises(PYACEMAKERError, match="Simulated Oracle failure"):
        oracle.run()
