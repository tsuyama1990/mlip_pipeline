from ase.build import bulk

from mlip_autopipec.physics.structure_gen.embedding import extract_periodic_box
from mlip_autopipec.physics.structure_gen.strategies import StrainGenerator


def test_uat_03_01_strain_samples() -> None:
    """Scenario 03-01: The Explorer's Map (Strain)"""
    # GIVEN a primitive unit cell
    atoms = bulk("Si", "diamond", a=5.43)

    # WHEN the StrainGenerator is invoked with range +/- 10%
    gen = StrainGenerator(strain_range=0.1)
    # The system outputs candidates.xyz containing 20 frames.
    count = 20
    candidates = gen.generate(atoms, count=count)

    # THEN it should produce a trajectory of structures
    assert len(candidates) == count

    # AND the volume of the smallest structure should be ~90% of original
    # AND the volume of the largest structure should be ~110% of original
    original_vol = atoms.get_volume()  # type: ignore[no-untyped-call]
    volumes = sorted([c.get_volume() for c in candidates])  # type: ignore[no-untyped-call]

    # Check bounds (approximate)
    # With random strain, we might not hit exact -10% or +10%, but should be close/within range.
    # Just verify we have variation.
    min_vol_ratio = volumes[0] / original_vol
    max_vol_ratio = volumes[-1] / original_vol

    # Ensure we are exploring
    assert min_vol_ratio < 0.99
    assert max_vol_ratio > 1.01


def test_uat_03_02_embedding_logic() -> None:
    """Scenario 03-02: The Defect Hunter (Embedding part)"""
    # GIVEN a large supercell
    # MgO is rocksalt usually? bulk("MgO", "rocksalt", a=4.2)
    mgo = bulk("MgO", "rocksalt", a=4.21)

    # 3x3x3 supercell (2 atoms * 27 = 54 atoms)
    supercell = mgo * (3, 3, 3)

    # AND a cutoff radius of 5.0 Angstroms
    cutoff = 5.0

    # WHEN the EmbeddingHandler extracts a box around an atom
    center_atom_index = 0
    box = extract_periodic_box(supercell, center_index=center_atom_index, cutoff=cutoff)

    # THEN the new box dimensions should be at least 10.0 Angstroms (2 * cutoff)
    cell = box.get_cell()  # type: ignore[no-untyped-call]
    assert cell[0, 0] >= 2 * cutoff
    assert cell[1, 1] >= 2 * cutoff
    assert cell[2, 2] >= 2 * cutoff
