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

    # We expect extract_cluster to return a Structure that is a cluster centered at atom_index
    # For now, we mock the logic or expect it to return a structure.
    # Since implementation isn't there, we just define expectations.

    cluster = manager.extract_cluster(s, center_atom_index=0, radius=3.0)

    assert isinstance(cluster, Structure)
    # The cluster should contain at least the center atom
    assert len(cluster.positions) >= 1
    # Check symbols
    assert cluster.symbols[0] == "Si"

def test_embed_cluster():
    s = Structure(
        symbols=["Si"],
        positions=np.array([[0, 0, 0]]),
        cell=np.eye(3) * 5,
        pbc=(False, False, False) # Cluster usually non-periodic initially or we make it periodic
    )

    manager = CandidateManager()
    embedded = manager.embed_cluster(s, vacuum=5.0)

    assert isinstance(embedded, Structure)
    assert all(embedded.pbc)
    # Check cell size increased
    assert embedded.cell[0,0] >= 10.0
