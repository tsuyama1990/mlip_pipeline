from typing import cast

from ase import Atoms


def embed_cluster(cluster: Atoms, vacuum: float) -> Atoms:
    """
    Embed a non-periodic cluster into a periodic box with vacuum padding.

    Args:
        cluster: The atomic cluster to embed.
        vacuum: The amount of vacuum padding (in Angstroms) to add around the cluster.

    Returns:
        Atoms: The embedded cluster in a periodic supercell.

    Raises:
        ValueError: If vacuum is non-positive or cluster is empty.
    """
    if vacuum <= 0:
        msg = "Vacuum must be positive."
        raise ValueError(msg)
    if len(cluster) == 0:
        msg = "Cluster is empty."
        raise ValueError(msg)

    # Create a copy to avoid modifying the input
    embedded = cluster.copy()  # type: ignore[no-untyped-call]

    # Reset cell and pbc (optional, center usually handles it, but good for cleanliness)
    embedded.set_cell([0.0, 0.0, 0.0])
    embedded.set_pbc(False)

    # Center the cluster and add vacuum
    # ase.Atoms.center sets the cell to span + 2*vacuum and centers positions
    embedded.center(vacuum=vacuum)

    # Set PBC to True for periodic calculation (e.g. QE needs it)
    embedded.set_pbc(True)

    return cast(Atoms, embedded)
