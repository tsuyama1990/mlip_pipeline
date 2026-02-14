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
        project=ProjectConfig(name="Test", root_dir=Path(".")),
        oracle=OracleConfig(
            dft=DFTConfig(
                code="qe", pseudopotentials={"Fe": "Fe.pbe.UPF"}, parameters={}
            ),
            mock=False,
        ),
    )


def test_dft_oracle_update_structure_logic(config: PYACEMAKERConfig) -> None:
    """Test the _update_structure_common method logic in isolation."""
    oracle = DFTOracle(config)
    s1 = StructureMetadata(tags=["test"])

    # mock result atoms
    result_atoms = MagicMock(spec=Atoms)
    result_atoms.get_potential_energy.return_value = -13.6
    # ASE methods return numpy arrays, we must mock this so .tolist() works
    result_atoms.get_forces.return_value = np.array([[0.0, 0.0, 0.0]])
    result_atoms.get_stress.return_value = np.array([0.0] * 6)

    oracle._update_structure_common(s1, result_atoms)

    assert s1.status == StructureStatus.CALCULATED
    assert s1.features["energy"] == -13.6
    assert s1.features["forces"] == [[0.0, 0.0, 0.0]]
    assert s1.features["atoms"] == result_atoms


def test_dft_oracle_compute_batch_flow(config: PYACEMAKERConfig) -> None:
    """Test DFTOracle compute_batch flow (chunking and skipping)."""
    oracle = DFTOracle(config)

    # Create structures
    atoms1 = Atoms("H")
    s1 = StructureMetadata(tags=["test"], features={"atoms": atoms1})
    s2 = StructureMetadata(tags=["test"]) # No atoms
    s3 = StructureMetadata(tags=["test"], status=StructureStatus.CALCULATED)

    structures = [s1, s2, s3]
    result_atoms = Atoms("H") # Placeholder result

    # Mock DFTManager to return iterator
    # Mock _update_structure_common to verify it is called
    with patch("pyacemaker.oracle.manager.DFTManager.compute_batch", return_value=iter([result_atoms])) as mock_compute:
        with patch.object(oracle, "_update_structure_common") as mock_update:
            results_iter = oracle.compute_batch(structures)
            results = list(results_iter)

            assert len(results) == 3
            # Check calling args
            mock_compute.assert_called_once()
            # Verify update called for s1
            mock_update.assert_called_once_with(s1, result_atoms)

            # s3 should be skipped (yielded as is)
            # Compare IDs because Pydantic equality might be strict or object identity issues in iterator
            assert results[2].id == s3.id
            assert results[2].status == StructureStatus.CALCULATED


def test_mock_oracle_simulation_failure(config: PYACEMAKERConfig) -> None:
    """Test MockOracle simulating a failure."""
    # Configure simulate_failure
    config.oracle.dft.parameters["simulate_failure"] = True

    oracle = MockOracle(config)

    with pytest.raises(PYACEMAKERError, match="Simulated Oracle failure"):
        oracle.run()


def test_mock_oracle_determinism(config: PYACEMAKERConfig) -> None:
    """Test that MockOracle produces deterministic results with seed."""
    config.oracle.dft.parameters["seed"] = 123
    oracle1 = MockOracle(config)

    s1 = StructureMetadata(tags=["test"])
    res1_iter = oracle1.compute_batch([s1])
    res1 = next(res1_iter)

    # Reset oracle with same seed
    oracle2 = MockOracle(config)
    s2 = StructureMetadata(tags=["test"])
    res2_iter = oracle2.compute_batch([s2])
    res2 = next(res2_iter)

    assert res1.features["energy"] == res2.features["energy"]
    assert res1.features["forces"] == res2.features["forces"]
