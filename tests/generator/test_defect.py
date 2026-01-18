import pytest
from ase.build import bulk

from mlip_autopipec.config.schemas.generator import (
    DefectConfig,
    DistortionConfig,
    GeneratorConfig,
    NMSConfig,
    SQSConfig,
)
from mlip_autopipec.generator.defect import DefectGenerator


@pytest.fixture
def defect_generator():
    return DefectGenerator()

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

def test_create_interstitial_different_element(defect_generator):
    atoms = bulk('Si', 'diamond', a=5.43) * (2,2,2) # 16 atoms

    interstitials = defect_generator.create_interstitial(atoms, element='H')

    # Should return at least 1 structure
    assert len(interstitials) > 0

    for inter in interstitials:
        assert len(inter) == 17 # One more atom
        assert inter.info['config_type'] == 'interstitial'
        assert inter.get_chemical_symbols().count('H') == 1

def test_defects_disabled():
    # To test "disabled", we need to test DefectApplicator, not Generator,
    # because Generator is now stateless and doesn't know about config.
    from mlip_autopipec.generator.defect import DefectApplicator

    config = GeneratorConfig(
        sqs=SQSConfig(),
        distortion=DistortionConfig(),
        nms=NMSConfig(),
        defects=DefectConfig(enabled=False)
    )

    applicator = DefectApplicator(config)
    atoms = bulk('Si')

    # Passing a list of atoms
    result = applicator.apply([atoms], 'Si')

    # Should just return original
    assert len(result) == 1
    assert result[0] == atoms
