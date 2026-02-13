import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

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
from mlip_autopipec.domain_models.enums import ExecutionMode
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import HaltInfo, Structure


def test_uat_cycle06_local_loop() -> None:
    print("Starting UAT Cycle 06: Local Learning Loop")  # noqa: T201

    # Setup Config
    config = GlobalConfig(
        orchestrator=OrchestratorConfig(work_dir=Path("uat_work_dir"), execution_mode=ExecutionMode.MOCK),
        generator=GeneratorConfig(),
        oracle=OracleConfig(),
        trainer=TrainerConfig(),
        dynamics=DynamicsConfig(),
        validator=ValidatorConfig(),
        active_learning=ActiveLearningConfig(n_candidates=5)
    )

    # Mock Dependencies
    generator = MagicMock()
    # Note: generator (BaseGenerator) is no longer used for local generation in ActiveLearner
    # because ActiveLearner uses internal CandidateGenerator.

    oracle = MagicMock()
    # Label them
    def compute_side_effect(structures: list[Structure]) -> list[Structure]:
        labeled = []
        for s in structures:
            s.label_status = "labeled"
            s.energy = -5.0
            labeled.append(s)
        return labeled
    oracle.compute.side_effect = compute_side_effect

    trainer = MagicMock()
    trainer.train.return_value = Potential(path=Path("new_pot.yace"), format="yace")
    # Select all - mock for internal ActiveSelector if we were mocking it,
    # but ActiveLearner uses internal ActiveSelector which uses config to decide.
    # We didn't inject selector, so it uses real one.
    # Real ActiveSelector uses ActiveSetMethod.RANDOM by default if not maxvol.
    # Since config uses default (NONE -> RANDOM/First N), it will select.

    # Instantiate ActiveLearner
    active_learner = ActiveLearner(config, generator, oracle, trainer)

    # Create Halt Event
    halt = HaltInfo(
        step=500,
        max_gamma=10.0,
        structure=Structure(atoms=Atoms("H2"), provenance="md_halt"),
        reason="high_uncertainty"
    )

    # Run Process
    print("Triggering ActiveLearner.process_halt...")  # noqa: T201
    new_potential = active_learner.process_halt(halt)

    # Verify
    assert str(new_potential.path) == "new_pot.yace"

    # Verify Oracle was called (meaning candidates were generated and selected)
    oracle.compute.assert_called_once()
    assert len(oracle.compute.call_args[0][0]) > 0

    # Verify Trainer was called
    trainer.train.assert_called_once()

    print("UAT Cycle 06 PASSED")  # noqa: T201

if __name__ == "__main__":
    test_uat_cycle06_local_loop()
