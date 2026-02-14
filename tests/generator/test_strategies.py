"""Tests for structure generation strategies."""

from typing import Any

import numpy as np
from ase.build import bulk

from pyacemaker.generator.strategies import DefectStrategy, M3GNetStrategy, RandomStrategy


def test_random_strategy() -> None:
    """Test random strategy generates diverse candidates."""
    atoms = bulk("Cu", "fcc", a=3.6)
    strategy = RandomStrategy(strain_range=0.1, rattle_amplitude=0.1)

    candidates = strategy.generate(atoms, n_candidates=5)

    assert len(candidates) == 5
    # Check candidates are different from seed
    for c in candidates:
        assert c is not atoms
        assert not (c.get_positions() == atoms.get_positions()).all()  # type: ignore[no-untyped-call]


def test_defect_strategy() -> None:
    """Test defect strategy creates vacancies."""
    # Create 2x2x2 supercell (32 atoms)
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    assert len(atoms) == 32

    strategy = DefectStrategy(defect_density=0.01)  # Should create 1 vacancy

    candidates = strategy.generate(atoms, n_candidates=1)

    assert len(candidates) == 1
    assert len(candidates[0]) == 31  # 32 - 1 vacancy


def test_defect_strategy_exhausted() -> None:
    """Test defect strategy stops if no atoms left."""
    atoms = bulk("Cu")  # 1 atom
    strategy = DefectStrategy(defect_density=2.0)  # Request removal of >1 atoms

    # max(1, round(1 * 2.0)) = 2 defects
    # 1st: remove 1 atom -> 0 atoms
    # 2nd: remove from 0 atoms -> no-op/break

    candidates = strategy.generate(atoms, n_candidates=1)
    assert len(candidates) == 1
    assert len(candidates[0]) == 0


def test_m3gnet_strategy_fallback(mocker: Any) -> None:
    """Test M3GNet strategy falls back when not installed."""
    # Mock fallback strategy
    mock_fallback = mocker.Mock(spec=RandomStrategy)
    atoms = bulk("Cu", "fcc", a=3.6)
    mock_fallback.generate.return_value = [atoms.copy()]  # type: ignore[no-untyped-call]

    # Force ImportError for m3gnet (or verify it's not installed)
    # Since we don't have m3gnet installed, it should trigger fallback logic
    # But if run in environment where it IS installed, we need to mock ImportError?
    # sys.modules patch can mask it?
    # mocker.patch.dict("sys.modules", {"m3gnet.models": None}) might trigger AttributeError or something else.
    # To force ImportError we can mock builtins.__import__ but that's risky.
    # Easiest is to rely on it not being installed.

    strategy = M3GNetStrategy(fallback_strategy=mock_fallback)

    candidates = strategy.generate(atoms, n_candidates=1)

    # Should call fallback
    mock_fallback.generate.assert_called_once()
    assert len(candidates) == 1


def test_m3gnet_strategy_success(mocker: Any) -> None:
    """Test M3GNet strategy success path."""
    # Mock Relaxer
    mock_relaxer_instance = mocker.Mock()
    # relax returns dict with 'final_structure'
    mock_relaxed_atoms = bulk("Cu", "fcc", a=3.7, cubic=True)  # Different a
    mock_relaxer_instance.relax.return_value = {"final_structure": mock_relaxed_atoms}

    # Mock Relaxer class
    mock_relaxer_cls = mocker.Mock(return_value=mock_relaxer_instance)

    # Patch sys.modules to simulate import success
    # We need to ensure m3gnet.models is a module object with Relaxer attribute
    mock_m3gnet_models = mocker.Mock()
    mock_m3gnet_models.Relaxer = mock_relaxer_cls
    mocker.patch.dict("sys.modules", {"m3gnet": mocker.Mock(), "m3gnet.models": mock_m3gnet_models})

    atoms = bulk("Cu", "fcc", a=3.6)
    strategy = M3GNetStrategy()

    candidates = strategy.generate(atoms, n_candidates=1)

    assert len(candidates) == 1
    # Check it used relaxer
    mock_relaxer_instance.relax.assert_called()
    # Check result is the mocked relaxed structure
    # Use index 0 for cell access [0,0]
    assert np.isclose(candidates[0].get_cell()[0, 0], 3.7)  # type: ignore[no-untyped-call]
