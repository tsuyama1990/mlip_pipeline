
import numpy as np

from mlip_autopipec.domain_models import Structure
from mlip_autopipec.infrastructure.mocks import MockOracle


def test_oracle_batch_compute() -> None:
    oracle = MockOracle()
    structures = [
        Structure(positions=np.array([[0.0, 0.0, 0.0]]), cell=np.eye(3), species=["H"])
        for _ in range(10)
    ]

    results = oracle.compute_batch(structures)
    assert len(results) == 10
    for res in results:
        assert res.energy is not None
        assert res.forces is not None

def test_large_dataset_simulation() -> None:
    # Simulate processing a large batch without OOM (conceptually, by checking we don't crash)
    oracle = MockOracle()
    # Create a generator of 1000 structures
    structures = [
        Structure(positions=np.array([[0.0, 0.0, 0.0]]), cell=np.eye(3), species=["H"])
        for _ in range(1000)
    ]
    results = oracle.compute_batch(structures)
    assert len(results) == 1000
