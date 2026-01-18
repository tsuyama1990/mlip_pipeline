import numpy as np
import pytest
from mlip_autopipec.surrogate.sampling import FPSSampler

def test_fps_sampler_initialization():
    sampler = FPSSampler()
    assert sampler is not None

def test_fps_selection_simple_1d():
    # Points on a line: 0, 1, 2, ..., 10
    features = np.array([[float(i)] for i in range(11)])
    sampler = FPSSampler()

    # Select 2 points. Should be endpoints (0 and 10)
    indices = sampler.select(features, n_samples=2)
    assert len(indices) == 2
    assert 0 in indices
    assert 10 in indices

    # Select 3 points. Should be 0, 10, 5
    indices = sampler.select(features, n_samples=3)
    assert len(indices) == 3
    assert 0 in indices
    assert 10 in indices
    assert 5 in indices

def test_fps_selection_random():
    # Random points in 3D
    np.random.seed(42)
    features = np.random.rand(100, 3)
    sampler = FPSSampler()

    indices = sampler.select(features, n_samples=10)
    assert len(indices) == 10
    assert len(set(indices)) == 10 # All unique

def test_fps_selection_scores():
    features = np.array([[0.0], [10.0], [5.0]])
    sampler = FPSSampler()

    # First point usually random or first.
    # FPSSampler logic: first point is the one with max distance to others?
    # Or usually strictly random first point or index 0?
    # Spec says "S = {s0} (random start)".
    # We should probably allow specifying seed or start for deterministic testing.

    # But usually FPS implementation picks the first one randomly or 0.
    # Let's verify we get scores.
    indices, scores = sampler.select_with_scores(features, n_samples=3)
    assert len(indices) == 3
    assert len(scores) == 3

def test_fps_more_samples_than_data():
    features = np.array([[0.0], [1.0]])
    sampler = FPSSampler()
    with pytest.raises(ValueError):
        sampler.select(features, n_samples=3)
