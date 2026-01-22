import numpy as np

from mlip_autopipec.surrogate.sampling import FarthestPointSampling


def test_fps_simple_2d():
    # Points: (0,0), (1,1), (0.1, 0.1), (10,10)
    descriptors = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.1, 0.1],
        [10.0, 10.0]
    ])

    # Assume seed 0 or first index ensures 0 is picked if we implement it that way.
    # The Spec says: "Initialize: Pick the first structure index i0 (e.g., the one with the lowest predicted energy)."
    # To make it deterministic for this test, we might need to supply energies or force start index.
    # If the implementation picks index 0 by default if no energies provided:

    fps = FarthestPointSampling(n_samples=2)
    # We might pass an optional 'energies' argument or just rely on index 0.
    indices = fps.select(descriptors)

    assert len(indices) == 2
    assert 0 in indices
    assert 3 in indices

def test_fps_n_samples_greater_than_data():
    descriptors = np.array([[0,0], [1,1]])
    fps = FarthestPointSampling(n_samples=5)
    indices = fps.select(descriptors)
    assert len(indices) == 2
    assert set(indices) == {0, 1}

def test_fps_n_samples_zero():
    descriptors = np.array([[0,0], [1,1]])
    fps = FarthestPointSampling(n_samples=0)
    indices = fps.select(descriptors)
    assert len(indices) == 0
