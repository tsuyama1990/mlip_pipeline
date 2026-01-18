import numpy as np
from ase import Atoms

from mlip_autopipec.inference.masking import ForceMasker


def test_force_masker_apply():
    # Create a cluster with atoms at specific distances
    # Center at 5,5,5
    center = np.array([5.0, 5.0, 5.0])

    # Atom 1: Center (dist 0) -> Mask 1
    # Atom 2: dist 2.0 (Core < 3.0) -> Mask 1
    # Atom 3: dist 3.5 (Core 3.0, Buffer 2.0) -> Mask 0
    # Atom 4: dist 6.0 (Outside) -> Mask 0

    atoms = Atoms("H4", positions=[
        [5.0, 5.0, 5.0],
        [7.0, 5.0, 5.0],
        [8.5, 5.0, 5.0],
        [11.0, 5.0, 5.0]
    ], cell=[20.0, 20.0, 20.0], pbc=True)

    masker = ForceMasker()
    masker.apply(atoms, center=center, radius=3.0)

    assert "force_mask" in atoms.arrays
    mask = atoms.arrays["force_mask"]

    assert mask[0] == 1.0
    assert mask[1] == 1.0
    assert mask[2] == 0.0
    assert mask[3] == 0.0

def test_force_masker_boundary():
    # Test atom exactly on boundary
    center = np.array([0.0, 0.0, 0.0])
    # 3 atoms
    atoms = Atoms("H3", positions=[
        [2.999999, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [3.000001, 0.0, 0.0]
    ], cell=[10.0, 10.0, 10.0], pbc=True)

    masker = ForceMasker()
    # Radius 3.0
    masker.apply(atoms, center=center, radius=3.0)
    mask = atoms.arrays["force_mask"]

    assert mask[0] == 1.0 # Inside
    assert mask[1] == 1.0 # On boundary (implementation is <=)
    assert mask[2] == 0.0 # Outside
