from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.structure_gen.embedding import EmbeddingHandler
import numpy as np

def test_extract_cluster():
    # Create a dummy structure: 1D chain for simplicity
    # 0.0, 2.0, 4.0, 6.0, 8.0
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    cell = np.eye(3) * 10.0
    struct = Structure(
        symbols=["Si"] * 5,
        positions=positions,
        cell=cell,
        pbc=(True, True, True)
    )

    # Extract around atom 2 (pos 4.0) with radius 2.1 (should include neighbors at 2.0 and 6.0)
    # Vacuum 5.0
    handler = EmbeddingHandler()
    cluster = handler.extract_cluster(struct, center_index=2, radius=2.1, vacuum=5.0)

    assert len(cluster.positions) == 3 # Center + 2 neighbors
    # Check vacuum padding: cluster cell should be large enough
    # Bounding box of 3 atoms is approx 4.0 wide. + 2*vacuum(5.0) = 14.0
    assert cluster.cell[0, 0] >= 10.0
    assert cluster.pbc == (True, True, True) # It should be wrapped in periodic box as per memory

    # Check that the central atom is roughly in the center of the new cell
    # This might depend on implementation, but usually centered.
