import pytest
from ase import Atoms

from mlip_autopipec.domain_models.config import ActiveLearningConfig, TrainerConfig
from mlip_autopipec.domain_models.datastructures import Structure
from mlip_autopipec.domain_models.enums import ActiveSetMethod
from mlip_autopipec.trainer.active_selector import ActiveSelector


@pytest.fixture
def candidates() -> list[Structure]:
    return [
        Structure(atoms=Atoms("H"), provenance=f"cand_{i}", label_status="unlabeled")
        for i in range(10)
    ]


def test_selector_limit(candidates: list[Structure]) -> None:
    active_config = ActiveLearningConfig()
    trainer_config = TrainerConfig(n_active_set_per_halt=3, active_set_method=ActiveSetMethod.RANDOM)

    selector = ActiveSelector(active_config, trainer_config)
    selected = list(selector.select_batch(candidates))

    assert len(selected) == 3
    # Provenance might change depending on implementation
    assert all("_active_" in s.provenance for s in selected)


def test_selector_empty_input() -> None:
    active_config = ActiveLearningConfig()
    trainer_config = TrainerConfig()
    selector = ActiveSelector(active_config, trainer_config)

    selected = list(selector.select_batch([]))
    assert len(selected) == 0
