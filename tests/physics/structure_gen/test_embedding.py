import pytest
import numpy as np
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.structure_gen.embedding import EmbeddingHandler

@pytest.fixture
def large_structure():
    # Create a 3x3x3 cubic grid of atoms
    positions = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                positions.append([x, y, z])
    return Structure(
        symbols=["H"] * 27,
        positions=np.array(positions),
        cell=np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]]),
        pbc=(True, True, True),
    )

def test_extract_cluster(large_structure):
    handler = EmbeddingHandler()

    # Extract cluster around atom 13 (center, at 1,1,1)
    # Radius 0.9 should capture just the center
    # Radius 1.1 should capture 6 neighbors

    # We rely on indices or geometric center.
    # Spec says "Take a cluster of atoms from a halted MD run".
    # Usually we pass a central atom index or the whole structure and let it decide.
    # Let's assume extract_cluster(structure, center_index, radius, vacuum)

    cluster = handler.extract_cluster(
        large_structure,
        center_index=13, # atom at (1,1,1)
        radius=1.1,
        vacuum=5.0
    )

    assert isinstance(cluster, Structure)
    # 1 center + 6 neighbors = 7 atoms
    assert len(cluster.positions) == 7

    # Check box size
    # Cluster span is roughly 2.0 (from 0 to 2 in coords).
    # Vacuum is 5.0 on each side -> 10.0 padding.
    # Box should be larger than 10.0.
    assert np.all(np.diag(cluster.cell) >= 10.0)

    # Center atom should be roughly in the middle of the new box
    # We can check relative positions are preserved
