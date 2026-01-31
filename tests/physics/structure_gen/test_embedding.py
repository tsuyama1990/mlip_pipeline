import numpy as np
import pytest
from ase.build import bulk

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.structure_gen.embedding import EmbeddingHandler


@pytest.fixture
def large_structure():
    # 4x4x4 supercell of Si
    atoms = bulk("Si", "diamond", a=5.43) * (4, 4, 4)
    return Structure.from_ase(atoms)


def test_embed_cluster(large_structure):
    handler = EmbeddingHandler()

    # Center atom index 0
    # Radius 4.0 A (should include nearest neighbors)
    cluster = handler.embed_cluster(
        large_structure, center_index=0, radius=4.0, vacuum=10.0
    )

    # Check it's a Structure
    assert isinstance(cluster, Structure)

    # Original Si-Si bond is 2.35 A.
    # Radius 4.0 should include 1st neighbor shell (4 atoms) and maybe 2nd (12 atoms at 3.84 A).
    # Total atoms > 1.
    assert len(cluster.positions) > 1

    # Check pbc is true (for QE usually we run periodic, even if vacuum padded, or we treat it as molecule in box)
    # The spec says "wrap it in a vacuum-padded periodic box".
    assert all(cluster.pbc)

    # Check cell size
    # Cell should be roughly 2*radius + vacuum? Or diameter + vacuum?
    # Actually, usually 2*radius is the cluster diameter. + vacuum on each side?
    # Let's just check it's large enough.
    assert np.all(np.diag(cluster.cell) >= 10.0)


def test_embedding_vacuum_padding(large_structure):
    handler = EmbeddingHandler()
    radius = 3.0
    vacuum = 5.0

    cluster = handler.embed_cluster(
        large_structure, center_index=0, radius=radius, vacuum=vacuum
    )

    # Estimate extent
    # At least one atom at 0,0,0
    # Box should be extent + vacuum
    # Since we don't know exact extent without calculation, we check the lower bound
    # Box size > vacuum
    assert np.all(np.diag(cluster.cell) >= vacuum)

    # Check orthogonality (off-diagonal elements are 0)
    # The current implementation creates a cubic box (np.eye(3) * length)
    cell = cluster.cell
    off_diagonals = cell - np.diag(np.diag(cell))
    assert np.allclose(off_diagonals, 0.0)
