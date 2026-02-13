import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mlip_autopipec.oracle.embedding import Embedding


def test_embedding_cut_cluster() -> None:
    # Create a large supercell
    # Si bulk, 5.43 A lattice constant. 4x4x4 supercell is ~21.7 A side.
    prim = bulk("Si", "diamond", a=5.43)
    supercell = prim * (4, 4, 4)

    # Target center atom (index 0 is usually at 0,0,0)
    center_idx = 0
    cutoff = 4.0  # Should include nearest neighbors (2.35 A) and maybe next nearest (3.84 A)
    buffer = 2.0

    embedder = Embedding(cutoff=cutoff, buffer=buffer)
    cluster = embedder.cut_cluster(supercell, center_idx)

    # Checks
    assert isinstance(cluster, Atoms)

    # 1. Box size check
    # Box should be at least 2 * (cutoff + buffer) = 12.0
    # Or just large enough to contain the cluster with vacuum.
    # The requirement says "The box size must be at least 2 * (cutoff + buffer) to avoid self-interaction images."
    min_box_size = 2 * (cutoff + buffer)
    assert np.all(cluster.cell.lengths() >= min_box_size)

    # 2. Atom count check
    # Nearest neighbors in diamond are 4 at d=sqrt(3)/4 * a ~= 2.35
    # Next nearest are 12 at d=a/sqrt(2) ~= 3.84
    # So cutoff 4.0 should include central atom + 4 NN + 12 NNN = 17 atoms.
    # Let's just check it's > 1 and <= total atoms
    assert len(cluster) > 1
    assert len(cluster) < len(supercell)

    # 3. Center atom should be approximately at the center of the new box
    # Or at least shifted so it is not at the edge if we want to avoid PBC issues with the cut.
    # But standard practice is often to center the cluster.
    # Let's check that all atoms are within the box boundaries [0, L]
    positions = cluster.get_positions()  # type: ignore[no-untyped-call]
    cell = cluster.get_cell()  # type: ignore[no-untyped-call]
    # Apply MIC or just check coordinate range if we assume 0..L
    assert np.all(positions >= 0)
    assert np.all(positions <= cell.diagonal())


def test_embedding_invalid_index() -> None:
    atoms = bulk("Si")
    embedder = Embedding(cutoff=3.0, buffer=1.0)
    # The current implementation of cut_cluster might not check index bounds explicitly,
    # but accessing atoms[index] would raise IndexError.
    # However, to be robust, we should check if the mock implementation raises it.
    # If the implementation uses neighbor lists, it might not fail immediately unless we check.
    # But usually atoms[100] on a 2-atom system will fail.

    # We need to make sure 'bulk' is valid. Si has 2 atoms.
    with pytest.raises(IndexError):
        embedder.cut_cluster(atoms, 100)
