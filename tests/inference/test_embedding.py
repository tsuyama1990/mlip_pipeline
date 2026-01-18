import numpy as np
import pytest
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

    positions = cluster.get_positions()
    # Check if one atom is at box_center
    dist_to_center = np.linalg.norm(positions - box_center, axis=1)
    min_dist = np.min(dist_to_center)
    assert min_dist < 1e-4

def test_embedding_pbc_crossing():
    # Test extraction across boundary
    atoms = bulk("Al", "fcc", a=4.0, cubic=True)
    atoms = atoms * (4, 4, 4) # 16x16x16 box

    # Atom at 0,0,0 (index 0)
    center_idx = 0

    config = EmbeddingConfig(core_radius=3.0, buffer_width=1.0) # Total 4.0

    extractor = EmbeddingExtractor(config)
    extracted = extractor.extract(atoms, center_idx)

    cluster = extracted.atoms
    # Verify we have neighbors
    assert len(cluster) > 1

def test_embedding_invalid_inputs():
    config = EmbeddingConfig()
    extractor = EmbeddingExtractor(config)

    # 1. Not atoms object
    with pytest.raises(TypeError):
        extractor.extract("not atoms", 0)

    # 2. Empty atoms
    with pytest.raises(ValueError):
        extractor.extract(Atoms(), 0)

    # 3. Invalid index
    atoms = Atoms("H1", positions=[[0,0,0]])
    with pytest.raises(IndexError):
        extractor.extract(atoms, 99)

    with pytest.raises(IndexError):
        extractor.extract(atoms, -2) # Negative index usually valid in python (-1), but strict check might raise if logic demands positive?
        # My implementation raises if < 0.
