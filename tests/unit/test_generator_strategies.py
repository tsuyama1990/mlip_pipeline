import numpy as np
import pytest
import json
from ase import Atoms
from ase.build import bulk

def test_apply_strain_hydrostatic():
    from mlip_autopipec.generator.transformations import apply_strain

    atoms = bulk("Cu", "fcc", a=3.6)
    vol_0 = atoms.get_volume()

    # 10% expansion
    strain = np.eye(3) * 0.1
    strained = apply_strain(atoms, strain)

    vol_1 = strained.get_volume()
    expected_vol = vol_0 * (1.1 ** 3)

    assert np.isclose(vol_1, expected_vol)
    assert strained.info["config_type"] == "strain"

    # Deserialize strain tensor
    stored_strain = json.loads(strained.info["strain_tensor"])
    assert np.allclose(stored_strain, strain)


def test_apply_rattle():
    from mlip_autopipec.generator.transformations import apply_rattle

    atoms = bulk("Cu", "fcc", a=3.6)
    pos_0 = atoms.get_positions()

    rattled = apply_rattle(atoms, sigma=0.1)
    pos_1 = rattled.get_positions()

    assert not np.allclose(pos_0, pos_1)
    diff = pos_1 - pos_0
    # check that mean displacement is roughly 0
    assert np.allclose(np.mean(diff, axis=0), 0, atol=0.2)
    assert rattled.info["config_type"] == "rattle"
    assert rattled.info["rattle_sigma"] == 0.1


def test_defect_vacancy():
    from mlip_autopipec.generator.defects import DefectStrategy
    from mlip_autopipec.config.schemas.generator import DefectConfig

    atoms = bulk("Cu", "fcc", cubic=True) * (2, 2, 2) # 32 atoms
    n_atoms = len(atoms)

    strategy = DefectStrategy(DefectConfig(enabled=True, vacancies=True))
    vacancies = strategy.generate_vacancies(atoms, count=1)

    assert len(vacancies) > 0
    for v in vacancies:
        assert len(v) == n_atoms - 1
        assert v.info["config_type"] == "vacancy"
        assert "defect_index" in v.info


def test_defect_interstitial():
    from mlip_autopipec.generator.defects import DefectStrategy
    from mlip_autopipec.config.schemas.generator import DefectConfig

    atoms = bulk("Fe", "bcc", cubic=True) # 2 atoms
    n_atoms = len(atoms)

    strategy = DefectStrategy(DefectConfig(enabled=True, interstitials=True, interstitial_elements=["H"]))
    interstitials = strategy.generate_interstitials(atoms)

    # BCC Fe cubic has 2 atoms. Threshold in generate_interstitials for Voronoi is 4 atoms.
    # So it uses fallback candidates.
    assert len(interstitials) > 0
    for i in interstitials:
        assert len(i) == n_atoms + 1
        assert i.get_chemical_symbols().count("H") == 1
        assert i.info["config_type"] == "interstitial"


def test_sqs_fallback_random(mocker):
    # Mock icet to fail import or be None
    mocker.patch.dict("sys.modules", {"icet": None})

    from mlip_autopipec.generator.sqs import SQSStrategy
    from mlip_autopipec.config.schemas.generator import SQSConfig

    prim = bulk("Fe")
    comp = {"Fe": 0.5, "Ni": 0.5}

    strategy = SQSStrategy(SQSConfig(enabled=True, supercell_size=[2, 2, 2]))

    sqs = strategy.generate(prim, comp)

    assert len(sqs) == len(prim) * 8
    syms = sqs.get_chemical_symbols()
    assert syms.count("Fe") == syms.count("Ni")
    assert sqs.info["origin"] == "random_shuffle"
