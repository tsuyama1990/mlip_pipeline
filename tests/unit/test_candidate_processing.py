import numpy as np
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.orchestration.candidate_processing import CandidateManager

def test_extract_cluster():
    # Create a dummy supercell
    s = Structure(
        symbols=["Si", "Si"],
        positions=np.array([[0, 0, 0], [5, 0, 0]]),
        cell=np.eye(3) * 10,
        pbc=(True, True, True)
    )

    manager = CandidateManager()

    # extract_cluster now wraps extract_periodic_box with box=2*radius
    cluster = manager.extract_cluster(s, center_atom_index=0, radius=3.0)

    assert isinstance(cluster, Structure)
    assert len(cluster.positions) >= 1

    # New expectation: it is periodic
    assert all(cluster.pbc)

    # Check cell size: 2 * 3.0 = 6.0
    assert np.isclose(cluster.cell[0,0], 6.0)

def test_embed_cluster_legacy():
    # Test legacy behavior for non-periodic input
    s = Structure(
        symbols=["Si"],
        positions=np.array([[0, 0, 0]]),
        cell=np.eye(3) * 5,
        pbc=(False, False, False)
    )

    manager = CandidateManager()
    embedded = manager.embed_cluster(s, vacuum=5.0)

    assert isinstance(embedded, Structure)
    assert all(embedded.pbc)
    # Check cell size increased (0->10 with vacuum centered, or existing cell + vacuum?)
    # Implementation: atoms.center(vacuum=5.0).
    # If atoms had cell, center() might change it.
    assert embedded.cell[0,0] >= 10.0

def test_embed_cluster_passthrough():
    # Test pass-through for already periodic input
    s = Structure(
        symbols=["Si"],
        positions=np.array([[0, 0, 0]]),
        cell=np.eye(3) * 5,
        pbc=(True, True, True)
    )

    manager = CandidateManager()
    embedded = manager.embed_cluster(s, vacuum=5.0)

    # Should be identical
    assert np.allclose(embedded.cell, s.cell)
    assert all(embedded.pbc)
