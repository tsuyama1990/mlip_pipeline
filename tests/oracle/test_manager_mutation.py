"""Tests for side effects in DFTManager."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.config import CONSTANTS, DFTConfig
from pyacemaker.oracle.manager import DFTManager


@pytest.fixture
def dft_config_mock(monkeypatch: pytest.MonkeyPatch) -> DFTConfig:
    """Mock DFT config."""
    monkeypatch.setattr(CONSTANTS, "skip_file_checks", True)
    return DFTConfig(
        pseudopotentials={"H": "H.upf"},
        embedding_enabled=True,
        embedding_buffer=2.0,
    )


def test_compute_no_mutation(dft_config_mock: DFTConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that compute does not mutate the input structure."""
    # Mock create_calculator to avoid running real DFT
    mock_calc = MagicMock()
    mock_calc.parameters = {"input_data": {"electrons": {}}}

    monkeypatch.setattr(
        "pyacemaker.oracle.manager.create_calculator", lambda *args, **kwargs: mock_calc
    )

    manager = DFTManager(dft_config_mock)
    original_atoms = Atoms("H", positions=[[0, 0, 0]])
    original_cell = original_atoms.get_cell().copy()  # type: ignore[no-untyped-call]

    # Run compute
    result = manager.compute(original_atoms)

    # Check original is untouched
    # Original was non-periodic and 0 cell
    assert not any(original_atoms.pbc)
    assert np.allclose(original_atoms.get_cell(), original_cell)  # type: ignore[no-untyped-call]

    # Check result is embedded (pbc=True)
    assert result is not None
    assert all(result.pbc)
    # Check result has different cell
    assert not np.allclose(result.get_cell(), original_cell)  # type: ignore[no-untyped-call]
