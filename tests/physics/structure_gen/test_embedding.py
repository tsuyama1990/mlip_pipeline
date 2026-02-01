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


def test_extract_periodic_box(large_structure):
    handler = EmbeddingHandler()

    box_length = 8.0
    # 4x4x4 Si is large enough (~20A)

    cluster = handler.extract_periodic_box(
        large_structure, center_index=0, box_length=box_length
    )

    assert isinstance(cluster, Structure)
    assert all(cluster.pbc)

    # Check cell is cubic box_length
    expected_cell = np.eye(3) * box_length
    assert np.allclose(cluster.cell, expected_cell)

    # Check center atom position
    # The center atom (index 0) should be at center of box (half_box, half_box, half_box)
    # But note: extract_periodic_box creates a new list of atoms. We assume the first one is likely the center
    # if it was index 0, but NeighborList order is not guaranteed to put center first unless we enforce it.
    # However, my implementation added `if not new_positions: ...` fallback.
    # Let's check if the center atom is actually in the list.
    # The center atom has distance 0.

    dists = cluster.arrays.get('cluster_dist')
    assert dists is not None
    assert 0.0 in dists

    # Check number of atoms > 1 (Si neighbors are at 2.35, so 8.0 box should catch them)
    assert len(cluster.positions) > 1

    # Verify strict containment
    # All positions should be within [0, box_length]
    pos = cluster.positions
    assert np.all(pos >= 0.0)
    assert np.all(pos <= box_length)
