import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.common import EmbeddingConfig
from mlip_autopipec.utils.embedding import EmbeddingExtractor


@pytest.fixture
def embedding_config():
    return EmbeddingConfig(core_radius=2.0, buffer_width=1.0)


def test_extract_basic(embedding_config):
    # Linear chain: 0 -- 1 -- 2 -- 3
    # Distance 2.0
    # Center at 1 [2,0,0].

    atoms = Atoms("H4", positions=[[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0]])
    extractor = EmbeddingExtractor(embedding_config)

    cluster = extractor.extract(atoms, 1)  # Center at [2,0,0]

    assert len(cluster) == 3  # 0, 1, 2

    # Check relative distances.
    # Center (index 0 in cluster) to others should be 2.0
    center_pos = cluster.positions[0]
    dists = np.linalg.norm(cluster.positions - center_pos, axis=1)
    dists = np.sort(dists)

    # Expected: 0.0 (self), 2.0, 2.0
    assert np.allclose(dists, [0.0, 2.0, 2.0])


def test_extract_pbc(embedding_config):
    # Atom at 0 and at 9. Box size 10.
    # Distance is 1.
    atoms = Atoms("H2", positions=[[0.1, 0, 0], [9.9, 0, 0]], cell=[10, 10, 10], pbc=True)

    extractor = EmbeddingExtractor(embedding_config)
    cluster = extractor.extract(atoms, 0)  # Center at 0.1

    assert len(cluster) == 2

    # Check distance
    dist = np.linalg.norm(cluster.positions[1] - cluster.positions[0])
    assert np.isclose(dist, 0.2)


def test_extract_invalid_input(embedding_config):
    extractor = EmbeddingExtractor(embedding_config)

    # Not atoms
    with pytest.raises(TypeError, match="Input must be an ase.Atoms object"):
        extractor.extract("not atoms", 0)  # type: ignore

    # Empty atoms
    with pytest.raises(ValueError, match="Input structure is empty"):
        extractor.extract(Atoms(), 0)

    # Index out of bounds
    atoms = Atoms("H", positions=[[0, 0, 0]])
    with pytest.raises(IndexError, match="out of bounds"):
        extractor.extract(atoms, 1)
