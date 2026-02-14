"""Tests for Oracle module."""

import secrets
from collections.abc import Iterator
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
def config(tmp_path: Path) -> PYACEMAKERConfig:
    """Return a valid configuration for DFTOracle."""
    # Create dummy pseudopotential file
    pp_file = tmp_path / "Fe.pbe.UPF"
    pp_file.touch()

    return PYACEMAKERConfig(
        version="0.1.0",
        project=ProjectConfig(name="Test", root_dir=tmp_path),
        oracle=OracleConfig(
            dft=DFTConfig(
                code="qe", pseudopotentials={"Fe": str(pp_file)}, parameters={}
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
    """Test DFTOracle compute_batch flow (streaming)."""
    oracle = DFTOracle(config)

    # Create structures
    atoms1 = Atoms("H")
    s1 = StructureMetadata(tags=["test"], features={"atoms": atoms1})
    s2 = StructureMetadata(tags=["test"])  # No atoms (should fail)
    # s3 is already calculated, so it must have energy/forces
    s3 = StructureMetadata(
        tags=["test"],
        status=StructureStatus.CALCULATED,
        energy=-5.0,
        forces=[[0.0, 0.0, 0.0]],
    )

    structures = [s1, s2, s3]
    result_atoms = Atoms("H")  # Placeholder result
    result_atoms.get_potential_energy = MagicMock(return_value=-10.0)  # type: ignore[assignment]
    result_atoms.get_forces = MagicMock(return_value=np.array([[0.0, 0.0, 0.0]]))  # type: ignore[assignment]

    # Mock DFTManager to return iterator
    # Mock _update_structure_common to verify it is called
    with (
        patch(
            "pyacemaker.modules.oracle.DFTManager.compute",
            return_value=result_atoms,
        ) as mock_compute,
        patch.object(
            oracle, "_update_structure_common", wraps=oracle._update_structure_common
        ) as mock_update,
    ):
        results_iter = oracle.compute_batch(structures)
        results = list(results_iter)

        assert len(results) == 3

        # Verify call count:
        # s1 -> compute called
        # s2 -> compute NOT called (no atoms)
        # s3 -> compute NOT called (already calculated)
        assert mock_compute.call_count == 1
        mock_compute.assert_called_with(atoms1)

        mock_update.assert_called_with(s1, result_atoms)

        # Check statuses
        assert results[0].status == StructureStatus.CALCULATED  # s1
        assert results[1].status == StructureStatus.FAILED  # s2
        assert results[2].status == StructureStatus.CALCULATED  # s3


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


def test_mock_oracle_randomness(config: PYACEMAKERConfig) -> None:
    """Test that MockOracle uses secure random if no seed provided."""
    # Ensure no seed
    if "seed" in config.oracle.dft.parameters:
        del config.oracle.dft.parameters["seed"]

    oracle = MockOracle(config)
    assert isinstance(oracle.rng, secrets.SystemRandom)


def test_dft_oracle_streaming_behavior(config: PYACEMAKERConfig) -> None:
    """Test that DFTOracle streams data (yields results incrementally)."""
    oracle = DFTOracle(config)

    # Create a generator that yields structures
    def structure_generator() -> Iterator[StructureMetadata]:
        for i in range(3):
            yield StructureMetadata(tags=[f"test_{i}"], features={"atoms": Atoms("H")})

    # Mock DFTManager to return dummy atoms
    dummy_atom = MagicMock(spec=Atoms)
    dummy_atom.get_potential_energy.return_value = -1.0
    dummy_atom.get_forces.return_value = np.array([[0.0, 0.0, 0.0]])
    dummy_atom.get_stress.return_value = np.array([0.0] * 6)

    # Use side_effect or return value
    with patch(
        "pyacemaker.modules.oracle.DFTManager.compute",
        return_value=dummy_atom,
    ) as mock_compute:
        results_iter = oracle.compute_batch(structure_generator())

        # Consume 1st
        r1 = next(results_iter)
        assert mock_compute.call_count == 1
        assert r1.status == StructureStatus.CALCULATED

        # Consume 2nd
        r2 = next(results_iter)
        assert mock_compute.call_count == 2
        assert r2.status == StructureStatus.CALCULATED

        # Consume 3rd
        r3 = next(results_iter)
        assert mock_compute.call_count == 3
        assert r3.status == StructureStatus.CALCULATED

        # Should be empty now
        with pytest.raises(StopIteration):
            next(results_iter)


def test_oracle_validation(config: PYACEMAKERConfig) -> None:
    """Test structure validation."""
    oracle = DFTOracle(config)
    # Using ignore because we are intentionally passing invalid type
    with pytest.raises(TypeError, match="Expected StructureMetadata"):
        next(oracle.compute_batch(["invalid"]))  # type: ignore[list-item]
