import pytest
from ase import Atoms
from ase.build import bulk

from mlip_autopipec.config.schemas.generator import DefectConfig
from mlip_autopipec.generator.defects import DefectStrategy


@pytest.fixture
def supercell() -> Atoms:
    # bulk returns primitive cell (1 atom for fcc) by default
    # repeat((2, 2, 2)) -> 1 * 8 = 8 atoms
    return bulk("Al", "fcc", a=4.05).repeat((2, 2, 2))


def test_defects_disabled_by_default(supercell: Atoms) -> None:
    config = DefectConfig(enabled=False)
    strategy = DefectStrategy(config)
    results = strategy.apply([supercell])
    assert len(results) == 1
    assert results[0] == supercell


def test_generate_vacancy_single(supercell: Atoms) -> None:
    # Strategy with vacancy enabled
    config = DefectConfig(enabled=True, vacancies=True, interstitials=False)
    strategy = DefectStrategy(config)

    # We call internal method directly to test logic
    vac_structures = strategy.generate_vacancies(supercell, count=1)

    # Expect 8 structures (one for each atom removed)
    assert len(vac_structures) == 8
    for s in vac_structures:
        assert len(s) == 7
        assert s.info["config_type"] == "vacancy"
        assert "defect_index" in s.info


def test_generate_vacancy_multiple(supercell: Atoms) -> None:
    config = DefectConfig(enabled=True, vacancies=True)
    strategy = DefectStrategy(config)

    # Remove 2 atoms
    vac_structures = strategy.generate_vacancies(supercell, count=2)
    # Should return 1 structure with 2 atoms removed
    assert len(vac_structures) == 1
    assert len(vac_structures[0]) == 6
    assert vac_structures[0].info["config_type"] == "vacancy"
    assert len(vac_structures[0].info["defect_indices"]) == 2


def test_generate_interstitials(supercell: Atoms) -> None:
    config = DefectConfig(enabled=True, interstitials=True, interstitial_elements=["H"])
    strategy = DefectStrategy(config)

    int_structures = strategy.generate_interstitials(supercell, "H")

    # Voronoi should find some sites.
    assert len(int_structures) > 0
    for s in int_structures:
        assert len(s) == 9  # 8 + 1
        assert s.symbols[-1] == "H"
        assert s.info["config_type"] == "interstitial"


def test_apply_strategy(supercell: Atoms) -> None:
    config = DefectConfig(
        enabled=True, vacancies=True, interstitials=True, interstitial_elements=["H"]
    )
    strategy = DefectStrategy(config)

    # Input list of 1 structure
    results = strategy.apply([supercell], primary_element="Al")

    # Output should include original + vacancies + interstitials
    # Original: 1
    # Vacancies (single): 8
    # Interstitials: > 0 (limited to 5 in implementation)
    n_int = len(strategy.generate_interstitials(supercell, "H"))
    expected_total = 1 + 8 + n_int
    assert len(results) == expected_total
