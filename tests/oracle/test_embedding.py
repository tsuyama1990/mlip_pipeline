"""Tests for periodic embedding logic in DFTManager."""

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.config import CONSTANTS, DFTConfig
from pyacemaker.oracle.manager import DFTManager


@pytest.fixture
def dft_config_embedding(monkeypatch: pytest.MonkeyPatch) -> DFTConfig:
    """DFT config with embedding enabled."""
    # Skip file checks for tests
    monkeypatch.setattr(CONSTANTS, "skip_file_checks", True)
    return DFTConfig(
        pseudopotentials={"H": "H.upf"},
        embedding_enabled=True,
        embedding_buffer=2.0,
    )


def test_embedding_non_periodic(dft_config_embedding: DFTConfig) -> None:
    """Test embedding applied to non-periodic structure."""
    manager = DFTManager(dft_config_embedding)
    # Simple molecule: H2 along X axis, length 1.0 A
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    assert not any(atoms.pbc)

    manager._apply_periodic_embedding(atoms)

    # Check PBC enabled
    assert all(atoms.pbc)

    # Check cell dimensions
    # Range X: 1.0. Buffer: 2.0 -> Total: 1.0 + 2*buffer = 5.0
    # Range Y, Z: 0.0. Buffer: 2.0 -> Total: 0.0 + 2*buffer = 4.0
    buffer = dft_config_embedding.embedding_buffer
    expected_x = 1.0 + 2 * buffer
    expected_yz = 0.0 + 2 * buffer

    cell = atoms.get_cell()  # type: ignore[no-untyped-call]
    assert np.isclose(cell[0, 0], expected_x)
    assert np.isclose(cell[1, 1], expected_yz)
    assert np.isclose(cell[2, 2], expected_yz)

    # Check centering: (0,0,0) -> should be shifted by buffer
    # Center of original: (0.5, 0, 0)
    # Center of new cell: (expected_x/2, expected_yz/2, expected_yz/2)
    # Expected Shift: buffer in all directions
    pos = atoms.get_positions()  # type: ignore[no-untyped-call]
    assert np.allclose(pos[0], [buffer, buffer, buffer])  # Was 0,0,0
    assert np.allclose(pos[1], [1.0 + buffer, buffer, buffer])  # Was 1,0,0


def test_embedding_already_periodic(dft_config_embedding: DFTConfig) -> None:
    """Test embedding skipped for already periodic structure."""
    manager = DFTManager(dft_config_embedding)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    original_cell = atoms.get_cell().copy()  # type: ignore[no-untyped-call]

    manager._apply_periodic_embedding(atoms)

    assert np.allclose(atoms.get_cell(), original_cell)  # type: ignore[no-untyped-call]
    assert all(atoms.pbc)


def test_embedding_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test embedding skipped when disabled in config."""
    monkeypatch.setattr(CONSTANTS, "skip_file_checks", True)
    config = DFTConfig(
        pseudopotentials={"H": "H.upf"},
        embedding_enabled=False,
    )
    manager = DFTManager(config)
    atoms = Atoms("H", positions=[[0, 0, 0]])

    manager._apply_periodic_embedding(atoms)

    assert not any(atoms.pbc)
    assert np.allclose(atoms.get_cell(), np.zeros((3, 3)))  # type: ignore[no-untyped-call]
