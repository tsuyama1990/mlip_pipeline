import pytest
import numpy as np

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.structure_gen.embedding import EmbeddingHandler


@pytest.fixture
def big_structure():
    # Create a 3x3x3 supercell of Si (approx 216 atoms)
    # Positions are not critical, just indices.
    # We will simulate extracting a central atom and its neighbors.
    positions = []
    symbols = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                positions.append([i*5.0, j*5.0, k*5.0])
                symbols.append("Si")

    return Structure(
        symbols=symbols,
        positions=np.array(positions),
        cell=np.eye(3) * 15.0,
        pbc=(True, True, True),
    )


def test_extract_cluster_simple(big_structure):
    # Extract atom 0 (at 0,0,0) and some neighbors?
    # Or just passing indices.
    indices = [0, 1, 2] # 0=(0,0,0), 1=(0,0,5), 2=(0,5,0)? No order depends on loop.
    # loop: k inner, j middle, i outer.
    # (0,0,0), (0,0,5), (0,0,10)...

    handler = EmbeddingHandler()
    cluster = handler.extract_cluster(big_structure, indices, vacuum=5.0)

    assert len(cluster.positions) == 3
    assert cluster.symbols == ["Si", "Si", "Si"]

    # Check cell size
    # Extracted positions:
    # 0: [0,0,0]
    # 1: [0,0,5]
    # 2: [0,0,10]
    # Span in Z is 10. Vacuum 5 on each side?
    # Usually vacuum is added to bounding box.
    # BBox: x=[0,0], y=[0,0], z=[0,10].
    # Cell size: x=vacuum*2? No, if span is 0.
    # Spec says "vacuum-padded periodic box".
    # Typically: cell = range + 2 * vacuum.

    # positions should be centered? Or shifted?
    # Usually we center the cluster in the new box.

    assert np.all(cluster.pbc)
    assert cluster.cell[0,0] >= 10.0 # 2*vacuum
    assert cluster.cell[2,2] >= 10.0 + 10.0 # range + 2*vacuum

def test_extract_cluster_pbc_handling():
    # If we extract atoms across PBC?
    # Embedding usually unwraps or assumes indices form a connected component.
    # If we just pick atoms, we assume they are the ones we want.
    # The handler should ideally recenter them to minimize bounding box.
    pass
