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

    # Create an infinite generator (or very large)
    def infinite_structures() -> Iterator[Atoms]:
        while True:
            yield Atoms("H")

    # Mock compute to return immediately
    with patch.object(manager, "compute") as mock_compute:
        mock_compute.side_effect = lambda s: s

        # Process only first 5 items from infinite stream
        batch_gen = manager.compute_batch(infinite_structures())

        results = []
        for i, res in enumerate(batch_gen):
            results.append(res)
            if i >= 4:
                break

        assert len(results) == 5
        assert mock_compute.call_count == 5
        # If it consumed more, this test would hang or call_count would be higher (if we could check async)
        # Being able to break the loop proves it yields lazily.


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
        with patch("ase.Atoms.get_potential_energy", side_effect=[Exception(error_msg), Exception(error_msg), -10.0]):
            manager.compute(atoms)

            assert mock_create.call_count == 3
            # Verify attempts passed to factory
            assert mock_create.call_args_list[0][0][1] == 0
            assert mock_create.call_args_list[1][0][1] == 1
            assert mock_create.call_args_list[2][0][1] == 2
