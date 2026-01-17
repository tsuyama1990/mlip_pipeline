import pytest
from ase.build import bulk

from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.generator.defect import DefectGenerator


@pytest.fixture
def defect_generator():
    config = GeneratorConfig()
    # Enable defects explicitly for test
    config.defects.enabled = True
    config.defects.vacancies = True
    config.defects.interstitials = True
    return DefectGenerator(config)

def test_create_vacancy(defect_generator):
    atoms = bulk('Si', 'diamond', a=5.43) * (2,2,2) # 16 atoms

    # Simple vacancy
    vacancies = defect_generator.create_vacancy(atoms)

    # Should return at least 1 structure (16 atoms -> 16 vacancies naive)
    assert len(vacancies) == 16

    for vac in vacancies:
        assert len(vac) == 15 # One less atom
        assert vac.info['config_type'] == 'vacancy'
        assert 'defect_index' in vac.info

def test_create_interstitial(defect_generator):
    atoms = bulk('Si', 'diamond', a=5.43) * (2,2,2) # 16 atoms

    interstitials = defect_generator.create_interstitial(atoms, element='Si')

    # Should return at least 1 structure
    assert len(interstitials) > 0

    for inter in interstitials:
        assert len(inter) == 17 # One more atom
        assert inter.info['config_type'] == 'interstitial'

def test_defects_disabled():
    config = GeneratorConfig()
    config.defects.enabled = False
    gen = DefectGenerator(config)
    atoms = bulk('Si')

    assert gen.create_vacancy(atoms) == []
    assert gen.create_interstitial(atoms, 'Si') == []
