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
            dft=DFTConfig(code="qe", pseudopotentials={"Fe": str(pp_file)}, parameters={}),
            mock=False,
        ),
    )


def test_dft_oracle_update_structure_logic(config: PYACEMAKERConfig) -> None:
    """Test the update_structure_metadata method logic in isolation."""
    from pyacemaker.core.utils import update_structure_metadata

    s1 = StructureMetadata(tags=["test"], features={"atoms": Atoms("H")})

    # mock result atoms
    result_atoms = MagicMock(spec=Atoms)
    result_atoms.get_potential_energy.return_value = -13.6
    # ASE methods return numpy arrays, we must mock this so .tolist() works
    result_atoms.get_forces.return_value = np.array([[0.0, 0.0, 0.0]])
    result_atoms.get_stress.return_value = np.array([0.0] * 6)

    update_structure_metadata(s1, result_atoms)

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
    result_atoms.get_potential_energy = MagicMock(return_value=-10.0)  # type: ignore[method-assign]
    result_atoms.get_forces = MagicMock(return_value=np.array([[0.0, 0.0, 0.0]]))  # type: ignore[method-assign]

    # Mock DFTManager to return iterator
    # Mock update_structure_metadata to verify it is called

    def side_effect_update(s: StructureMetadata, atoms: Atoms | None) -> None:
        if atoms:
            s.status = StructureStatus.CALCULATED
            s.energy = -10.0

    with (
        patch(
            "pyacemaker.modules.oracle.DFTManager.compute",
            return_value=result_atoms,
        ) as mock_compute,
        patch(
            "pyacemaker.modules.oracle.update_structure_metadata",
            side_effect=side_effect_update
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

        # Check statuses (Order independent checks)
        # s1 should be CALCULATED
        assert s1.status == StructureStatus.CALCULATED
        # s2 should be FAILED
        assert s2.status == StructureStatus.FAILED
        # s3 should be CALCULATED
        assert s3.status == StructureStatus.CALCULATED

        # Verify values for s1
        assert s1.energy == -10.0


def test_mock_oracle_simulation_failure(config: PYACEMAKERConfig) -> None:
    """Test MockOracle simulating a failure."""
    # Configure simulate_failure
    config.oracle.dft.parameters["simulate_failure"] = True

    oracle = MockOracle(config)

    with pytest.raises(PYACEMAKERError, match="Simulated Oracle failure"):
        oracle.run()


def test_mock_oracle_determinism(config: PYACEMAKERConfig) -> None:
    """Test that MockOracle produces deterministic results with seed."""
    # Initialize config parameters first
    config.oracle.dft.parameters["seed"] = 123

    # Create oracle1
    oracle1 = MockOracle(config)
    s1 = StructureMetadata(tags=["test"])
    # Need atoms for validation pass inside compute_batch logic for MockOracle too?
    # MockOracle creates dummy atoms if missing.
    res1_iter = oracle1.compute_batch([s1])
    res1 = next(res1_iter)

    # Create oracle2
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
    # Accessing private attribute strictly for test verification
    assert isinstance(oracle.rng, secrets.SystemRandom)


def test_dft_oracle_streaming_behavior(config: PYACEMAKERConfig) -> None:
    """Test that DFTOracle streams data (yields results incrementally)."""
    # Reduce workers to control buffer size (buffer = workers * 2 = 2)
    config.oracle.dft.max_workers = 1
    config.oracle.dft.chunk_size = 2  # No longer used but kept for config validity
    oracle = DFTOracle(config)

    # Create a generator that yields structures
    def structure_generator() -> Iterator[StructureMetadata]:
        # Yield more structures to verify we don't process all of them
        for i in range(10):
            yield StructureMetadata(tags=[f"test_{i}"], features={"atoms": Atoms("H")})

    # Mock DFTManager to return dummy atoms
    dummy_atom = MagicMock(spec=Atoms)
    dummy_atom.get_potential_energy.return_value = -1.0
    dummy_atom.get_forces.return_value = np.array([[0.0, 0.0, 0.0]])
    dummy_atom.get_stress.return_value = np.array([0.0] * 6)

    # Use side_effect or return value
    # We patch update_structure_metadata to prevent actual calculation status update failing due to mock mismatch
    with patch(
        "pyacemaker.modules.oracle.DFTManager.compute",
        return_value=dummy_atom,
    ) as mock_compute, patch(
        "pyacemaker.modules.oracle.update_structure_metadata"
    ) as mock_update:
        # We need update_structure_metadata to set status=CALCULATED manually if mocked out,
        # or we verify calls. The original implementation sets s.status=CALCULATED inside update.
        # Let's side effect the mock to set status.
        def update_side_effect(s: StructureMetadata, a: Atoms | None) -> None:
            s.status = StructureStatus.CALCULATED

        mock_update.side_effect = update_side_effect

        results_iter = oracle.compute_batch(structure_generator())

        # Initial check - nothing processed yet (generator not consumed)
        assert mock_compute.call_count == 0

        # Consume 1st item
        r1 = next(results_iter)
        assert r1.status == StructureStatus.CALCULATED

        # Consume 2nd item
        r2 = next(results_iter)
        assert r2.status == StructureStatus.CALCULATED

        # With sliding window and background workers, call count is at least 2
        # but strictly less than 10 (proving streaming)
        # Typically 2 or 3 depending on race conditions
        assert mock_compute.call_count >= 2
        assert mock_compute.call_count < 10

        # Consume 3rd item
        r3 = next(results_iter)
        assert r3.status == StructureStatus.CALCULATED

        # Verify we still haven't processed everything
        assert mock_compute.call_count >= 3
        assert mock_compute.call_count < 10

        # We don't need to consume the rest. The test proves:
        # 1. We get results (r1, r2, r3)
        # 2. We don't submit all 10 tasks at once (call_count < 10)


def test_oracle_validation(config: PYACEMAKERConfig) -> None:
    """Test structure validation."""
    oracle = DFTOracle(config)
    # Using ignore because we are intentionally passing invalid type
    with pytest.raises(TypeError, match="Expected StructureMetadata"):
        next(oracle.compute_batch(["invalid"]))  # type: ignore[list-item]
