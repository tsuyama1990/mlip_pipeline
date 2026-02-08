import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule

from mlip_autopipec.components.oracle.embedding import embed_cluster
from mlip_autopipec.constants import MAX_VACUUM_SIZE


def test_embed_cluster_basic() -> None:
    """Test embedding a simple cluster in a vacuum box."""
    cluster = molecule("H2")  # type: ignore[no-untyped-call]
    # Keep copy of original positions
    original_positions = cluster.get_positions()  # type: ignore[no-untyped-call]
    original_cell = cluster.get_cell()  # type: ignore[no-untyped-call]

    vacuum = 5.0
    embedded = embed_cluster(cluster, vacuum)

    # Check PBC
    assert np.all(embedded.pbc)

    # Check cell dimensions
    # Assuming the cluster is centered, the cell should be > cluster size + 2 * vacuum
    positions = embedded.get_positions()  # type: ignore[no-untyped-call]
    # molecule H2 is small, ~0.74A
    # span is small
    cluster_span = np.ptp(positions, axis=0)
    cell_diag = np.diag(embedded.get_cell())  # type: ignore[no-untyped-call]

    # The cell size is usually calculated as span + 2*vacuum
    assert np.all(cell_diag >= cluster_span + 2 * vacuum - 1e-5)

    # Check centering (approximate)
    cell_center = cell_diag / 2.0
    # center() usually centers the geometry center, mean might be slightly off for asymmetric molecules but H2 is symmetric
    # Using center of geometry (bounding box center)
    bbox_center = (np.min(positions, axis=0) + np.max(positions, axis=0)) / 2
    assert np.allclose(bbox_center, cell_center, atol=1e-5)

    # Verify input was not modified
    assert np.allclose(cluster.get_positions(), original_positions)  # type: ignore[no-untyped-call]
    assert np.allclose(cluster.get_cell(), original_cell)  # type: ignore[no-untyped-call]


def test_embed_cluster_from_bulk() -> None:
    """Test extracting a cluster from bulk and embedding it."""
    # Create a 4x4x4 bulk Si supercell
    si_bulk = bulk("Si", "diamond", a=5.43) * (4, 4, 4)

    # Extract a cluster (e.g. atoms within 6A of center)
    center = si_bulk.get_cell().sum(axis=0) / 2  # type: ignore[no-untyped-call]
    # Ensure unwrapped positions if needed, but bulk usually fits in cell
    distances = np.linalg.norm(si_bulk.positions - center, axis=1)
    mask = distances < 6.0

    cluster = si_bulk[mask]

    # Embed
    vacuum = 4.0
    embedded = embed_cluster(cluster, vacuum)

    assert len(embedded) == len(cluster)
    assert np.all(embedded.pbc)

    # Check cell size
    cell_diag = np.diag(embedded.get_cell())  # type: ignore[no-untyped-call]
    positions = embedded.get_positions()  # type: ignore[no-untyped-call]
    cluster_extent = np.ptp(positions, axis=0)
    assert np.all(cell_diag >= cluster_extent + 2 * vacuum - 1e-5)


def test_embed_cluster_invalid_vacuum() -> None:
    """Test invalid vacuum value."""
    cluster = molecule("H2")  # type: ignore[no-untyped-call]
    with pytest.raises(ValueError, match="Vacuum must be positive"):
        embed_cluster(cluster, -1.0)


def test_embed_cluster_vacuum_too_large() -> None:
    """Test vacuum size limit check."""
    cluster = molecule("H2")  # type: ignore[no-untyped-call]
    with pytest.raises(ValueError, match="Vacuum exceeds"):
        embed_cluster(cluster, MAX_VACUUM_SIZE + 1.0)


def test_embed_cluster_empty() -> None:
    """Test empty cluster."""
    cluster = Atoms()
    with pytest.raises(ValueError, match="Cluster is empty"):
        embed_cluster(cluster, 5.0)
