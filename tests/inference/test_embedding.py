import numpy as np
from ase import Atoms
from ase.build import bulk

from mlip_autopipec.config.schemas.inference import EmbeddingConfig
from mlip_autopipec.inference.embedding import EmbeddingExtractor


def test_embedding_extractor_fcc():
    # Create a 5x5x5 Al supercell
    # Al lattice constant ~ 4.05
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms = atoms * (5, 5, 5)

    # Center atom index (approx middle)
    # 5x5x5 = 125 atoms.
    center_idx = 62

    # Config: Core=4.0, Buffer=2.0. Box size ~ 12.0
    config = EmbeddingConfig(core_radius=4.0, buffer_width=2.0)

    extractor = EmbeddingExtractor(config)
    extracted = extractor.extract(atoms, center_idx)

    # Verification
    # 1. Extracted object type
    assert extracted.origin_index == center_idx
    # 2. Check atoms
    cluster = extracted.atoms
    assert isinstance(cluster, Atoms)
    assert len(cluster) > 0
    assert len(cluster) < len(atoms)

    # 3. Check central atom is at center of box
    # Box size is 2*(4+2) = 12.0
    box_center = np.array([6.0, 6.0, 6.0])
    # The focal atom in the new cluster should be close to box_center
    # But which one is the focal atom?
    # Usually the first one or we track it.
    # The spec says "Center the focal atom at (L/2, L/2, L/2)".
    # Let's assume the extractor handles shifting.
    # We need to find the focal atom in the new cluster.
    # Maybe ExtractedStructure should tell us the new index?
    # Or we assume it's index 0? The spec doesn't specify.
    # Let's assume it puts it at index 0 or we check positions.

    # Ideally, we find the atom closest to box_center
    distances = cluster.get_distances(range(len(cluster)), range(len(cluster)), mic=True)
    # This is n x n matrix.

    positions = cluster.get_positions()
    # Check if one atom is at box_center
    dist_to_center = np.linalg.norm(positions - box_center, axis=1)
    min_dist = np.min(dist_to_center)
    assert min_dist < 1e-4

    # 4. Check neighbors
    # In original fcc, nearest neighbor distance is 4.05 / sqrt(2) ~ 2.86
    # Core radius 4.0 should include 1st shell (12 atoms).
    # Buffer 2.0 (total 6.0) should include 2nd shell (a=4.05), 3rd shell?
    # 1st shell: 2.86
    # 2nd shell: 4.05
    # 3rd shell: 4.95
    # 4th shell: 5.72
    # Radius 6.0 should capture up to 4th shell.

    # Let's just verify that atoms within cut+buffer are included.
    # And distances are preserved.

def test_embedding_pbc_crossing():
    # Test extraction across boundary
    atoms = bulk("Al", "fcc", a=4.0, cubic=True)
    atoms = atoms * (4, 4, 4) # 16x16x16 box

    # Atom at 0,0,0 (index 0)
    center_idx = 0

    config = EmbeddingConfig(core_radius=3.0, buffer_width=1.0) # Total 4.0
    # Neighbor at -2.8 (wrapped to 13.2) should be included if distance < 4.0

    extractor = EmbeddingExtractor(config)
    extracted = extractor.extract(atoms, center_idx)

    cluster = extracted.atoms
    # Verify we have neighbors
    assert len(cluster) > 1
