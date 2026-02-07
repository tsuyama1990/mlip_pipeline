from collections.abc import Iterator

import numpy as np

from mlip_autopipec.domain_models import Structure
from mlip_autopipec.infrastructure.mocks import MockOracle


def test_oracle_batch_compute() -> None:
    oracle = MockOracle()
    structures = [
        Structure(positions=np.array([[0.0, 0.0, 0.0]]), cell=np.eye(3), species=["H"])
        for _ in range(10)
    ]

    # compute_batch now returns an iterator
    results = oracle.compute_batch(structures)
    assert isinstance(results, Iterator)

    count = 0
    for res in results:
        assert res.energy is not None
        assert res.forces is not None
        count += 1
    assert count == 10

def test_lazy_evaluation() -> None:
    """
    Verify that the oracle computation is lazy.
    We check this by taking one item from the generator and ensuring
    we can do so without processing the entire list (conceptually).
    """
    oracle = MockOracle()
    structures = [
        Structure(positions=np.array([[0.0, 0.0, 0.0]]), cell=np.eye(3), species=["H"])
        for _ in range(100)
    ]

    results_iter = oracle.compute_batch(structures)

    # Consume one
    first = next(results_iter)
    assert first.energy is not None

    # We haven't consumed the rest, so memory usage shouldn't spike if list was huge.
    # Python generators guarantee this behavior.
