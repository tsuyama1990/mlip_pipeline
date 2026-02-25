"""Tests for Active Learner module."""

from collections.abc import Iterator
from unittest.mock import patch

import pytest

from pyacemaker.domain_models.models import StructureMetadata, StructureStatus, UncertaintyState
from pyacemaker.modules.active_learner import ActiveLearner


def generate_mock_structures(count: int) -> Iterator[StructureMetadata]:
    """Generate mock structures with varying uncertainty."""
    for i in range(count):
        uncertainty = 0.1 * i  # Increasing uncertainty
        yield StructureMetadata(
            uncertainty_state=UncertaintyState(gamma_max=uncertainty, gamma_mean=uncertainty),
            tags=["mock"],
            status=StructureStatus.NEW,
        )


def test_select_batch_top_k() -> None:
    """Test selection of top K structures by uncertainty."""
    learner = ActiveLearner()
    # 10 structures with uncertainty 0.0 to 0.9
    candidates = generate_mock_structures(10)

    # Select top 3
    selected = learner.select_batch(candidates, n_select=3)

    assert len(selected) == 3
    # Top 3 should be 0.9, 0.8, 0.7. Order matters (descending)
    uncertainties = [
        s.uncertainty_state.gamma_max
        for s in selected
        if s.uncertainty_state and s.uncertainty_state.gamma_max is not None
    ]
    assert sorted(uncertainties, reverse=True) == pytest.approx([0.9, 0.8, 0.7])


def test_select_batch_threshold() -> None:
    """Test filtering by uncertainty threshold."""
    learner = ActiveLearner()
    candidates = generate_mock_structures(10)  # 0.0 to 0.9

    # Select top 5 but filter > 0.75
    # Only 0.9 and 0.8 satisfy > 0.75.
    selected = learner.select_batch(candidates, n_select=5, threshold=0.75)

    # Should select 0.9 and 0.8
    assert len(selected) == 2
    for s in selected:
        assert s.uncertainty_state is not None
        assert s.uncertainty_state.gamma_max is not None
        assert s.uncertainty_state.gamma_max > 0.75


def test_select_batch_memory_efficiency() -> None:
    """Verify heapq.nlargest is used for memory efficiency."""
    learner = ActiveLearner()
    candidates = generate_mock_structures(10)

    with patch("heapq.nlargest") as mock_nlargest:
        # Mock return value must be a list
        mock_nlargest.return_value = []
        learner.select_batch(candidates, n_select=5)
        mock_nlargest.assert_called_once()


def test_select_batch_edge_cases() -> None:
    """Test edge cases."""
    learner = ActiveLearner()
    candidates = generate_mock_structures(5)

    assert learner.select_batch(candidates, n_select=0) == []

    # Missing uncertainty
    s_no_unc = StructureMetadata(tags=["test"], status=StructureStatus.NEW)
    candidates_with_none = [s_no_unc]

    selected = learner.select_batch(candidates_with_none, n_select=1)
    # Should be selected but with -1.0 score (lowest priority if others exist, but here only one)
    assert len(selected) == 1
    assert selected[0] == s_no_unc
