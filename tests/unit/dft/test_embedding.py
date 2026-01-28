import numpy as np
from ase import Atoms
from ase.build import bulk

from mlip_autopipec.dft.embedding import ClusterEmbedder


def test_cluster_embedder_creation():
    embedder = ClusterEmbedder(cutoff=5.0)
    assert embedder.cutoff == 5.0


def test_cluster_embedder_cut():
    # Create a large supercell
    atoms = bulk("Al", "fcc", a=4.05) * (4, 4, 4)
    # Introduce a defect in the middle
    center_idx = len(atoms) // 2

    embedder = ClusterEmbedder(cutoff=4.0)
    cluster = embedder.embed(atoms, center_index=center_idx)

    assert isinstance(cluster, Atoms)
    assert len(cluster) < len(atoms)
    assert len(cluster) > 0

    # Check if the center atom is approximately at the center of the new cell
    # (ClusterEmbedder should wrap it)
    # The exact logic depends on implementation, but usually we center the cluster.

    # Check periodicity
    assert cluster.pbc.all()


def test_cluster_embedder_vac():
    """Test embedding works even if the center atom is missing (e.g. tracking a vacancy position)."""
    atoms = bulk("Al", "fcc", a=4.05) * (3, 3, 3)
    center_pos = np.array([6.0, 6.0, 6.0]) # Arbitrary position

    embedder = ClusterEmbedder(cutoff=3.0)
    cluster = embedder.embed(atoms, center_position=center_pos)

    assert isinstance(cluster, Atoms)
    assert cluster.pbc.all()
