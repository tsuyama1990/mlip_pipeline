from unittest.mock import patch

import numpy as np
import pytest
from ase.build import bulk

from mlip_autopipec.config.schemas.generator import GeneratorConfig

# We will implement AlloyGenerator in the next step, so we assume its interface.
# If importing fails, we might need to create a dummy file first or structure tests to run after implementation.
# But for TDD, we write the test first.
# To avoid ImportErrors during "Phase 2" verification (if I run tests before implementation),
# I'll create the file `mlip_autopipec/generator/alloy.py` with empty class or mock it?
# The instructions say "Create all source and test files" then "Generate Log".
# But "Verification & Proof of Work" says "Execute pytest immediately after generating the implementation file".
# So I can write the test now, and it will fail or I can write the implementation later.
# However, to be able to import in test, the file must exist.
from mlip_autopipec.generator.alloy import AlloyGenerator


@pytest.fixture
def alloy_generator():
    config = GeneratorConfig()
    return AlloyGenerator(config)

def test_sqs_generation_stoichiometry(alloy_generator):
    # Target: Fe50Ni50, 8 atoms
    # We remove the patch because we implemented fallback logic in AlloyGenerator.
    # The fallback uses random shuffle, which guarantees stoichiometry if we construct the symbol list correctly.

    prim = bulk('Fe', 'fcc', a=3.5)
    composition = {"Fe": 0.5, "Ni": 0.5}

    # Generate
    # We need small supercell to match test expectations easily
    # AlloyGenerator uses config.supercell_matrix
    # Default is [[2,0,0], [0,2,0], [0,0,2]] which is 2x2x2 = 8 atoms for 1-atom prim cell.

    sqs = alloy_generator.generate_sqs(prim, composition)

    assert len(sqs) == 8
    assert sqs.get_chemical_symbols().count('Fe') == 4
    assert sqs.get_chemical_symbols().count('Ni') == 4
    # Check fallback info
    assert sqs.info['config_type'] == 'sqs'
    assert sqs.info.get('origin') == 'random_shuffle'

def test_apply_strain(alloy_generator):
    atoms = bulk('Cu', 'fcc', a=3.6)
    vol_orig = atoms.get_volume()

    # Hydrostatic strain +10% volume means scale factor = (1.1)^(1/3)
    # But SPEC says "strain_range" is linear strain?
    # Spec says: "strain_range: Tuple[float, float] = (-0.05, 0.05)"
    # Usually strain is epsilon. L = L0 * (1 + epsilon). V approx V0 * (1 + 3*epsilon).
    # "Implement apply_strain(atoms, strain_tensor)."

    strain_tensor = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]) # 10% strain on all axes

    strained = alloy_generator.apply_strain(atoms, strain_tensor)

    # Volume should be roughly (1.1)^3 * original if strain is defined as 1+e
    # If strain tensor e means (I + e), then V_new = det(I+e) * V_old
    # det(1.1 * I) = 1.1^3 = 1.331

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

    # Check statistics (roughly)
    # With new implementation using np.random.normal(0, sigma), std should be close to sigma.
    # 1 atom is still small sample.

    atoms_large = atoms * (5,5,5)
    rattled_large = alloy_generator.apply_rattle(atoms_large, sigma=0.1)
    disps = rattled_large.get_positions() - atoms_large.get_positions()

    # Check standard deviation of the displacements matches sigma
    # Note: np.std calculates standard deviation. For N(0, sigma), sample std -> sigma.
    # We increase tolerance slightly for randomness.
    assert np.isclose(np.std(disps), 0.1, rtol=0.2)

    assert rattled_large.info['config_type'] == 'rattle'
    assert rattled_large.info['rattle_sigma'] == 0.1
