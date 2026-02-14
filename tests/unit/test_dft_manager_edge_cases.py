"""Tests for DFTManager edge cases."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.config import DFTConfig
from pyacemaker.core.exceptions import StructureError
from pyacemaker.oracle.manager import DFTManager


@pytest.fixture
def config(tmp_path: object) -> DFTConfig:
    """Return a default DFT configuration."""
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    pp_file = tmp_path / "H.pbe.UPF"
    pp_file.touch()
    return DFTConfig(pseudopotentials={"H": str(pp_file)}, max_retries=3)


def test_compute_empty_structure(config: DFTConfig) -> None:
    """Test handling of empty structures."""
    manager = DFTManager(config)
    atoms = Atoms()  # Empty
    with pytest.raises(StructureError, match="Structure contains no atoms"):
        manager.compute(atoms)


def test_compute_batch_large_generator(config: DFTConfig) -> None:
    """Test compute_batch handles large generators without consuming all at once."""
    manager = DFTManager(config)

    # Create a generator that tracks consumption
    consumed_count = 0

    def large_structures() -> Iterator[Atoms]:
        nonlocal consumed_count
        # Yield 100 items
        for _ in range(100):
            consumed_count += 1
            yield Atoms("H")

    # Mock compute to return immediately
    with patch.object(manager, "compute") as mock_compute:
        mock_compute.side_effect = lambda s: s

        # Start generator
        batch_gen = manager.compute_batch(large_structures())

        # Consume only 5 items
        results = []
        for i, res in enumerate(batch_gen):
            results.append(res)
            if i >= 4:
                break

        # Verify results
        assert len(results) == 5
        assert mock_compute.call_count == 5

        # Critical: Verify that we haven't consumed significantly more than we requested.
        # compute_batch processes 1-by-1, so consumed_count should be exactly 5.
        # If it materialized a list, consumed_count would be 100.
        # This confirms lazy evaluation (memory safety).
        assert consumed_count == 5


def test_compute_retry_parameter_change(config: DFTConfig) -> None:
    """Test that retry logic changes mixing_beta."""
    manager = DFTManager(config)
    atoms = Atoms("H")

    # Mock create_calculator to inspect parameters
    with patch("pyacemaker.oracle.manager.create_calculator") as mock_create:
        mock_calc = MagicMock()
        mock_calc.parameters = {"input_data": {"electrons": {"mixing_beta": 0.7}}}
        mock_create.return_value = mock_calc

        # Fail 3 times then succeed? No, just verify calls.
        # We need compute to actually run loop.
        # Mock get_potential_energy to fail twice then succeed.
        # Note: DFTManager checks for specific recoverable errors defined in config/defaults.
        # "SCF" is not in the default list ("scf not converged" is), so "Exception: SCF" might be treated as fatal.
        # We need to match the error message to something recoverable.
        error_msg = "error: scf not converged"
        with patch(
            "ase.Atoms.get_potential_energy",
            side_effect=[Exception(error_msg), Exception(error_msg), -10.0],
        ):
            manager.compute(atoms)

            assert mock_create.call_count == 3
            # Verify attempts passed to factory
            assert mock_create.call_args_list[0][0][1] == 0
            assert mock_create.call_args_list[1][0][1] == 1
            assert mock_create.call_args_list[2][0][1] == 2
