import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.inference import EmbeddingConfig
from mlip_autopipec.inference.embedding import EmbeddingExtractor


@pytest.fixture
def extractor():
    # Box size will be 2 * (2.0 + 1.0) = 6.0
    config = EmbeddingConfig(core_radius=2.0, buffer_width=1.0)
    return EmbeddingExtractor(config)


def test_extract_simple(extractor):
    # 2 atoms, distance 1.0. Core=2.0. Should include both.
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10], pbc=True)
    extracted = extractor.extract(atoms, 0)

    assert len(extracted.atoms) == 2
    assert extracted.origin_index == 0
    # Check shift: Center should be at [3.0, 3.0, 3.0] (half of 6.0)
    pos = extracted.atoms.positions[0]
    assert np.allclose(pos, [3.0, 3.0, 3.0])


def test_extract_pbc(extractor):
    # 2 atoms at boundary. 0 at 0.1, 1 at 9.9. Distance 0.2.
    atoms = Atoms("H2", positions=[[0.1, 0, 0], [9.9, 0, 0]], cell=[10, 10, 10], pbc=True)
    extracted = extractor.extract(atoms, 0)

    assert len(extracted.atoms) == 2
    # Verify relative position in new box
    # Atom 0 at [3,3,3]. Atom 1 should be at [2.8, 3, 3] approx
    # distance is 0.2
    d = extracted.atoms.get_distance(0, 1)
    assert abs(d - 0.2) < 1e-5


def test_extract_invalid_index(extractor):
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    with pytest.raises(IndexError):
        extractor.extract(atoms, 99)


def test_extract_empty(extractor):
    atoms = Atoms()
    with pytest.raises(ValueError, match="empty"):
        extractor.extract(atoms, 0)


def test_extract_type_error(extractor):
    with pytest.raises(TypeError):
        extractor.extract("not atoms", 0)  # type: ignore
