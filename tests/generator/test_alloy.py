import numpy as np
import pytest
from ase.build import bulk

from mlip_autopipec.config.schemas.common import Composition
from mlip_autopipec.config.schemas.generator import (
    DefectConfig,
    DistortionConfig,
    GeneratorConfig,
    NMSConfig,
    SQSConfig,
)
from mlip_autopipec.generator.alloy import AlloyGenerator


@pytest.fixture
def alloy_generator():
    config = GeneratorConfig(
        sqs=SQSConfig(enabled=True),
        distortion=DistortionConfig(enabled=True),
        nms=NMSConfig(),
        defects=DefectConfig()
    )
    return AlloyGenerator(config)

def test_sqs_generation_stoichiometry(alloy_generator):
    # Target: Fe50Ni50, 8 atoms
    prim = bulk('Fe', 'fcc', a=3.5)
    composition = Composition({"Fe": 0.5, "Ni": 0.5})

    # Generate
    # AlloyGenerator uses config.sqs.supercell_matrix

    sqs = alloy_generator.generate_sqs(prim, composition)

    assert len(sqs) == 8
    assert sqs.get_chemical_symbols().count('Fe') == 4
    assert sqs.get_chemical_symbols().count('Ni') == 4
    # Check fallback info
    assert sqs.info['config_type'] == 'sqs'
    assert sqs.info.get('origin') == 'random_shuffle'

def test_sqs_generation_invalid_composition(alloy_generator):
    prim = bulk('Fe', 'fcc', a=3.5)

    # We must construct validation manually or expect validation error on Composition creation.
    # However, if we pass a valid Composition but it somehow fails check inside, we test here.
    # But Composition validates itself.
    # Let's verify Composition raises ValueError
    with pytest.raises(ValueError):
        Composition({"Fe": 0.5, "Ni": 0.6})

def test_apply_strain(alloy_generator):
    atoms = bulk('Cu', 'fcc', a=3.6)
    vol_orig = atoms.get_volume()

    strain_tensor = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]) # 10% strain on all axes

    strained = alloy_generator.apply_strain(atoms, strain_tensor)

    expected_vol = vol_orig * (1.1 ** 3)
    assert np.isclose(strained.get_volume(), expected_vol)
    assert strained.info['config_type'] == 'strain'
    assert 'strain_tensor' in strained.info

def test_apply_rattle(alloy_generator):
    atoms = bulk('Cu', 'fcc', a=3.6)
    positions_orig = atoms.get_positions().copy()

    rattled = alloy_generator.apply_rattle(atoms, sigma=0.1)

    positions_new = rattled.get_positions()
    displacements = positions_new - positions_orig

    # Check that positions changed
    assert not np.allclose(positions_orig, positions_new)

    # Check cell didn't change
    assert np.allclose(atoms.get_cell(), rattled.get_cell())

    atoms_large = atoms * (5,5,5)
    rattled_large = alloy_generator.apply_rattle(atoms_large, sigma=0.1)
    disps = rattled_large.get_positions() - atoms_large.get_positions()

    assert np.isclose(np.std(disps), 0.1, rtol=0.2)

    assert rattled_large.info['config_type'] == 'rattle'
    assert rattled_large.info['rattle_sigma'] == 0.1

def test_generate_batch_disabled_distortions():
    config = GeneratorConfig(
        sqs=SQSConfig(enabled=True),
        distortion=DistortionConfig(enabled=False),
        nms=NMSConfig(),
        defects=DefectConfig()
    )
    gen = AlloyGenerator(config)

    base = bulk('Fe')
    batch = gen.generate_batch(base)

    assert len(batch) == 1
    assert batch[0] == base

def test_generate_batch_single_strain_step():
    # Edge case: n_strain_steps = 1
    config = GeneratorConfig(
        sqs=SQSConfig(),
        distortion=DistortionConfig(
            enabled=True,
            strain_range=(-0.05, 0.05),
            n_strain_steps=1,
            n_rattle_steps=1
        ),
        nms=NMSConfig(),
        defects=DefectConfig()
    )
    gen = AlloyGenerator(config)
    base = bulk('Fe')

    batch = gen.generate_batch(base)
    # Linspace(-0.05, 0.05, 1) -> [-0.05]. One strain.
    # Base + 1 strain = 2 structures.
    # 1 rattle per structure (including base?)
    # generate_batch logic:
    # results = [base]
    # + strain list -> results = [base, strained]
    # For s in [base, strained]: rattle 1 time.
    # total = 2 (base+strained) + 2 (rattled base + rattled strained) = 4 structures.

    assert len(batch) == 4
    types = [s.info.get('config_type') for s in batch]
    assert types.count('base') == 1
    assert types.count('strain') == 1
    assert types.count('rattle') == 2
