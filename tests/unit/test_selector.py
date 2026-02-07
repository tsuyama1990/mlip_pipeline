import numpy as np

from mlip_autopipec.domain_models import Structure
from mlip_autopipec.infrastructure.mocks import MockSelector


def test_mock_selector() -> None:
    # Setup
    candidates = []
    for _ in range(10):
        s = Structure(
            positions=np.array([[0.0, 0.0, 0.0]]),
            cell=np.eye(3),
            species=["H"]
        )
        candidates.append(s)

    # Init
    selector = MockSelector()

    # Run
    selected = selector.select(candidates, n=3)

    # Check
    assert len(selected) == 3
    for s in selected:
        assert isinstance(s, Structure)

def test_mock_selector_fewer_candidates() -> None:
    candidates = [
        Structure(positions=np.array([[0.0, 0.0, 0.0]]), cell=np.eye(3), species=["H"])
    ]
    selector = MockSelector()
    selected = selector.select(candidates, n=3)
    assert len(selected) == 1
