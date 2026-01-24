import logging

from ase.build import bulk

from mlip_autopipec.config.schemas.generator import DefectConfig
from mlip_autopipec.generator.defects import DefectStrategy

logging.basicConfig(level=logging.DEBUG)

def test_vacancy():
    # Use smaller supercell to avoid memory issues (Scalability)
    supercell = bulk("Al", "fcc", a=4.05).repeat((2, 2, 2))

    # Data Integrity Check
    if len(supercell) < 1:
        raise ValueError("Supercell is empty")

    print(f"Supercell size: {len(supercell)}")

    config = DefectConfig(enabled=True, vacancies=True)
    strategy = DefectStrategy(config)

    # Use iterator or small count
    vac_structures = strategy.generate_vacancies(supercell, count=1)
    print(f"Generated vacancies: {len(vac_structures)}")

if __name__ == "__main__":
    test_vacancy()
