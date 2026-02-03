import numpy as np
from ase.build import bulk

from mlip_autopipec.physics.structure_gen.embedding import extract_periodic_box


def test_extract_periodic_box_geometry() -> None:
    # 1. Setup a large supercell
    # Use cubic=True for consistent density checks
    prim = bulk("Cu", "fcc", a=3.6, cubic=True)
    supercell = prim * (5, 5, 5)

    # 2. Extract box with commensurate size
    # cutoff=1.8 -> box=3.6 (1 unit cell)
    cutoff = 1.8
    box = extract_periodic_box(supercell, center_index=0, cutoff=cutoff)

    # 3. Checks
    # a) Orthogonal cell?
    cell = box.get_cell()  # type: ignore[no-untyped-call]
    assert np.allclose(cell[0, 1], 0)
    assert np.allclose(cell[0, 2], 0)
    assert np.allclose(cell[1, 0], 0)
    assert np.allclose(cell[1, 2], 0)
    assert np.allclose(cell[2, 0], 0)
    assert np.allclose(cell[2, 1], 0)

    # b) Sufficiently large?
    assert cell[0, 0] >= 2 * cutoff
    assert cell[1, 1] >= 2 * cutoff
    assert cell[2, 2] >= 2 * cutoff

    # c) Atom count density
    # Should be exactly 1 unit cell (4 atoms)
    assert len(box) == 4

    # Try larger box
    # cutoff=3.6 -> box=7.2 (2x2x2 = 8 unit cells -> 32 atoms)
    box2 = extract_periodic_box(supercell, center_index=0, cutoff=3.6)
    assert len(box2) == 32
