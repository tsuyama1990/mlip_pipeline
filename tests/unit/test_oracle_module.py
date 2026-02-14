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
    assert s1.energy == -13.6
    assert s1.forces == [[0.0, 0.0, 0.0]]
    assert s1.features["atoms"] == result_atoms


def test_dft_oracle_compute_batch_flow(config: PYACEMAKERConfig) -> None:
    """Test DFTOracle compute_batch flow (chunking and skipping)."""
    oracle = DFTOracle(config)

    # Create structures
    atoms1 = Atoms("H")
    s1 = StructureMetadata(tags=["test"], features={"atoms": atoms1})
    s2 = StructureMetadata(tags=["test"])  # No atoms
    s3 = StructureMetadata(tags=["test"], status=StructureStatus.CALCULATED)

    structures = [s1, s2, s3]
    result_atoms = Atoms("H")  # Placeholder result

    # Mock DFTManager to return iterator
    # Mock _update_structure_common to verify it is called
    with (
        patch(
            "pyacemaker.modules.oracle.DFTManager.compute_batch",
            return_value=iter([result_atoms]),
        ) as mock_compute,
        patch.object(oracle, "_update_structure_common") as mock_update,
    ):
        results_iter = oracle.compute_batch(structures)
        results = list(results_iter)

        # Check calling args
        # Only s1 has atoms, so compute_batch should be called once with a list containing atoms1
        # Wait, DFTManager.compute_batch takes a list of atoms.
        # oracle.compute_batch processes chunks.
        # s1 has atoms. s2 has no atoms (skipped/failed internally or kept in buffer?).
        # s3 is already calculated.
        # Logic:
        # s1: atoms extracted -> added to chunk
        # s2: atoms None -> skipped adding to chunk?
        # s3: yielded immediately.
        # Chunk flushed at end.

        # Let's verify expectations based on implementation
        # s2 extract_atoms returns None (logged warning).
        # s1 is processed.
        # compute_batch called with [atoms1].
        # update called for s1.

        assert len(results) == 3
        mock_compute.assert_called()
        mock_update.assert_called_with(s1, result_atoms)

        # s3 should be yielded as is
        assert results[0].id == s3.id or results[1].id == s3.id or results[2].id == s3.id


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

    assert res1.energy == res2.energy
    assert res1.forces == res2.forces
