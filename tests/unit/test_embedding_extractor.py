
import pytest
import numpy as np
from ase import Atoms
from mlip_autopipec.config.schemas.inference import EmbeddingConfig
from mlip_autopipec.inference.embedding import EmbeddingExtractor

def test_embedding_extractor_initialization():
    config = EmbeddingConfig(core_radius=4.0, buffer_width=2.0)
    extractor = EmbeddingExtractor(config)
    assert extractor.config.core_radius == 4.0

def test_extract_cluster_simple():
    """
    Test extraction on a simple cubic lattice.
    """
    # Create a 3x3x3 supercell of Cu (fcc) -> but let's make it simple cubic
    # to easily count neighbors.
    # a=2.0. Center at (1,1,1) (middle of 3x3x3 grid).
    # Nearest neighbors distance = 2.0.

    positions = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                positions.append([x*2.0, y*2.0, z*2.0])

    atoms = Atoms('H27', positions=positions, cell=[6.0, 6.0, 6.0], pbc=True)

    # Center atom is at x=1, y=1, z=1 (index 13 in flattened loop: 1*9 + 1*3 + 1 = 13)
    center_idx = 13

    # We want cutoff=2.5 to include 6 neighbors (dist 2.0) but exclude others (dist sqrt(8) ~ 2.82)
    # core_radius + buffer_width = 2.5
    config = EmbeddingConfig(core_radius=1.5, buffer_width=1.0)
    extractor = EmbeddingExtractor(config)

    cluster = extractor.extract(atoms, center_idx)

    # Expected: Center + 6 neighbors = 7 atoms
    assert len(cluster) == 7
    assert not cluster.pbc.any()

    # Verify vacuum
    assert np.all(cluster.cell.lengths() > 10.0)

def test_extract_cluster_pbc():
    """
    Test extraction crossing PBC boundaries.
    """
    # 2 atoms in a small cell.
    # Cell = [10, 10, 10]
    # Atom A at [0.1, 0, 0]
    # Atom B at [9.9, 0, 0] -> Distance 0.2 via PBC.

    atoms = Atoms('H2', positions=[[0.1, 0, 0], [9.9, 0, 0]], cell=[10, 10, 10], pbc=True)

    # cutoff > 0.2
    config = EmbeddingConfig(core_radius=0.5, buffer_width=0.5)
    extractor = EmbeddingExtractor(config)

    # Extract around index 0. Should include index 1.
    cluster = extractor.extract(atoms, 0)

    assert len(cluster) == 2

    # Verify relative distance in cluster is small (~0.2), not large (~9.8)
    dists = cluster.get_all_distances()
    # dists is 2x2. dists[0,1]
    d = dists[0, 1]
    assert pytest.approx(d, 0.01) == 0.2

def test_input_validation():
    config = EmbeddingConfig()
    extractor = EmbeddingExtractor(config)

    with pytest.raises(TypeError):
        extractor.extract("not atoms", 0) # type: ignore

    atoms = Atoms('H')
    with pytest.raises(IndexError):
        extractor.extract(atoms, 99)
