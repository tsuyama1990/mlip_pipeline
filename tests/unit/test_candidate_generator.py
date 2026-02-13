
import pytest
from ase import Atoms

from mlip_autopipec.domain_models.config import ActiveLearningConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.generator.candidate_generator import CandidateGenerator


@pytest.fixture
def seed_structure() -> Structure:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10], pbc=True)
    return Structure(atoms=atoms, provenance="seed", label_status="labeled")


def test_candidate_generator_count(seed_structure: Structure) -> None:
    config = ActiveLearningConfig(n_candidates=5)
    generator = CandidateGenerator(config)

    candidates = list(generator.generate_local(seed_structure))
    assert len(candidates) == 5

    for cand in candidates:
        assert "_local_" in cand.provenance
        assert cand.label_status == "unlabeled"


def test_perturbation_applied(seed_structure: Structure) -> None:
    # Set magnitude large enough to be sure, or rely on random
    # But random might be 0? Unlikely.
    config = ActiveLearningConfig(perturbation_magnitude=0.1, n_candidates=1)
    generator = CandidateGenerator(config)

    candidate = next(generator.generate_local(seed_structure))

    # Check if positions changed
    assert (candidate.atoms.positions != seed_structure.atoms.positions).any()

    # Check max displacement (approx)
    # This is hard to test deterministically without seeding RNG
    # But we can check it's not IDENTICAL


def test_consistent_cell(seed_structure: Structure) -> None:
    config = ActiveLearningConfig()
    generator = CandidateGenerator(config)

    candidate = next(generator.generate_local(seed_structure))
    assert (candidate.atoms.cell == seed_structure.atoms.cell).all()
    assert (candidate.atoms.pbc == seed_structure.atoms.pbc).all()
