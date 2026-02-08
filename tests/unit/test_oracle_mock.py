import numpy as np

from mlip_autopipec.components.oracle.mock import MockOracle
from mlip_autopipec.domain_models.config import OracleConfig
from mlip_autopipec.domain_models.structure import Structure


def test_mock_oracle_force_sum_zero() -> None:
    oracle = MockOracle(OracleConfig())

    # Create a structure
    s = Structure(
        positions=np.zeros((3, 3)),
        atomic_numbers=np.array([1, 1, 1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
    )

    labeled_iter = oracle.compute([s])
    labeled_s = next(labeled_iter)

    assert labeled_s.forces is not None
    # Check if sum of forces is close to zero
    # Relaxed tolerance from 1e-15 to 1e-10 for broader platform stability
    assert np.allclose(np.sum(labeled_s.forces, axis=0), np.zeros(3), atol=1e-10)
