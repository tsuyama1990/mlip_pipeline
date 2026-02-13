from collections.abc import Iterable, Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from ase import Atoms

from mlip_autopipec.core.active_learner import ActiveLearner
from mlip_autopipec.domain_models.config import (
    ActiveLearningConfig,
    DynamicsConfig,
    GeneratorConfig,
    GlobalConfig,
    OracleConfig,
    OrchestratorConfig,
    TrainerConfig,
    ValidatorConfig,
)
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import HaltInfo, Structure


# Mocks for external components
class MockOracle:
    def compute(self, structures: Iterable[Structure]) -> Iterable[Structure]:
        out = []
        for s in structures:
            s.label_status = "labeled"
            s.energy = -1.0
            out.append(s)
        return out


class MockTrainer:
    def train(self, structures: Iterable[Structure], initial_potential: Potential | None = None) -> Potential:
        return Potential(path=Path("mock_path.yace"), format="yace")


# Mocks for internal components
class MockCandidateGenerator:
    def generate_local(self, structure: Structure) -> Iterator[Structure]:
        # Return 5 candidates
        return iter([Structure(atoms=Atoms("He"), provenance="local") for _ in range(5)])


class MockActiveSelector:
    def select_batch(self, candidates: Iterable[Structure]) -> Iterator[Structure]:
        # Select first 3
        return iter(list(candidates)[:3])


@pytest.fixture
def config(tmp_path: Path) -> GlobalConfig:
    # Use real configs with minimal valid values
    return GlobalConfig(
        orchestrator=OrchestratorConfig(work_dir=tmp_path),
        generator=GeneratorConfig(),
        oracle=OracleConfig(),
        trainer=TrainerConfig(),
        dynamics=DynamicsConfig(),
        validator=ValidatorConfig(),
        active_learning=ActiveLearningConfig()
    )


@pytest.fixture
def active_learner(config: GlobalConfig) -> ActiveLearner:
    # Dependencies
    gen = MagicMock() # BaseGenerator
    oracle = MockOracle()
    trainer = MockTrainer()
    cand_gen = MockCandidateGenerator()
    selector = MockActiveSelector()

    return ActiveLearner(
        config=config,
        generator=gen,
        oracle=oracle,  # type: ignore[arg-type]
        trainer=trainer, # type: ignore[arg-type]
        candidate_generator=cand_gen, # type: ignore[arg-type]
        active_selector=selector # type: ignore[arg-type]
    )


def test_process_halt(active_learner: ActiveLearner) -> None:
    halt_event = HaltInfo(
        step=100,
        max_gamma=10.0,
        structure=Structure(atoms=Atoms("He"), provenance="halt"),
        reason="high_uncertainty"
    )
    current_pot = Potential(path=Path("init.yace"), format="yace")

    new_potential = active_learner.process_halt(halt_event, current_pot)

    assert isinstance(new_potential, Potential)
    assert str(new_potential.path) == "mock_path.yace"


def test_process_halt_flow(active_learner: ActiveLearner) -> None:
    halt_event = HaltInfo(
        step=100,
        max_gamma=10.0,
        structure=Structure(atoms=Atoms("He"), provenance="halt"),
        reason="high_uncertainty"
    )
    current_pot = Potential(path=Path("init.yace"), format="yace")

    # Spy on trainer.train
    # We need to access the trainer on the active_learner instance.
    # Since we passed a MockTrainer, we can modify it.
    active_learner.trainer.train = MagicMock(return_value=Potential(path=Path("spy.yace"), format="yace"))  # type: ignore[method-assign]

    res = active_learner.process_halt(halt_event, current_pot)

    active_learner.trainer.train.assert_called_once()
    args, kwargs = active_learner.trainer.train.call_args
    # Check that initial_potential was passed
    assert kwargs.get('initial_potential') == current_pot

    assert str(res.path) == "spy.yace"


def test_process_halt_empty_selection(active_learner: ActiveLearner) -> None:
    """Test scenario where no candidates are selected."""
    # Override active_selector to return empty iterator
    active_learner.active_selector.select_batch = MagicMock(return_value=iter([])) # type: ignore[method-assign]

    halt_event = HaltInfo(
        step=100,
        max_gamma=10.0,
        structure=Structure(atoms=Atoms("He"), provenance="halt"),
        reason="high_uncertainty"
    )
    current_pot = Potential(path=Path("init.yace"), format="yace")

    # Mock trainer to ensure it is NOT called
    active_learner.trainer.train = MagicMock() # type: ignore[method-assign]

    res = active_learner.process_halt(halt_event, current_pot)

    # Should return current potential
    assert res == current_pot
    # Trainer should NOT be called
    active_learner.trainer.train.assert_not_called()
