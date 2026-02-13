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
from mlip_autopipec.domain_models.datastructures import HaltInfo, Potential, Structure


# Mocks for external components
class MockOracle:
    def compute(self, structures):
        for s in structures:
            s.label_status = "labeled"
            s.energy = -1.0
        return structures

class MockTrainer:
    def train(self, structures):
        return Potential(path=Path("mock_path.yace"), format="yace")

# Mocks for internal components
class MockCandidateGenerator:
    def generate_local(self, structure):
        # Return 5 candidates
        return [Structure(atoms=Atoms("He"), provenance="local") for _ in range(5)]

class MockActiveSelector:
    def select_batch(self, candidates):
        # Select first 3
        return list(candidates)[:3]

@pytest.fixture
def config(tmp_path):
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
def active_learner(config):
    # Dependencies
    gen = MagicMock() # BaseGenerator
    oracle = MockOracle()
    trainer = MockTrainer()
    cand_gen = MockCandidateGenerator()
    selector = MockActiveSelector()

    return ActiveLearner(
        config=config,
        generator=gen,
        oracle=oracle,
        trainer=trainer,
        candidate_generator=cand_gen,
        active_selector=selector
    )

def test_process_halt(active_learner):
    halt_event = HaltInfo(
        step=100,
        max_gamma=10.0,
        structure=Structure(atoms=Atoms("He"), provenance="halt"),
        reason="high_uncertainty"
    )

    new_potential = active_learner.process_halt(halt_event)

    assert isinstance(new_potential, Potential)
    assert str(new_potential.path) == "mock_path.yace"

def test_process_halt_flow(active_learner):
    halt_event = HaltInfo(
        step=100,
        max_gamma=10.0,
        structure=Structure(atoms=Atoms("He"), provenance="halt"),
        reason="high_uncertainty"
    )

    # Spy on trainer.train
    active_learner.trainer.train = MagicMock(return_value=Potential(path=Path("spy.yace"), format="yace"))

    res = active_learner.process_halt(halt_event)

    active_learner.trainer.train.assert_called_once()
    assert str(res.path) == "spy.yace"
