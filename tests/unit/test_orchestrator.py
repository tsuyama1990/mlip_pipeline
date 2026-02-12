from pathlib import Path

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.config import (
    DynamicsConfig,
    GeneratorConfig,
    GlobalConfig,
    OracleConfig,
    OrchestratorConfig,
    TrainerConfig,
    ValidatorConfig,
)
from mlip_autopipec.domain_models.enums import (
    DynamicsType,
    ExecutionMode,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)


def test_orchestrator_init(tmp_path: Path) -> None:
    config = GlobalConfig(
        orchestrator=OrchestratorConfig(work_dir=tmp_path),
        generator=GeneratorConfig(type=GeneratorType.MOCK),
        oracle=OracleConfig(type=OracleType.MOCK),
        trainer=TrainerConfig(type=TrainerType.MOCK),
        dynamics=DynamicsConfig(type=DynamicsType.MOCK),
        validator=ValidatorConfig(type=ValidatorType.MOCK),
    )
    orch = Orchestrator(config)
    assert orch.work_dir == tmp_path
    assert orch.state_manager is not None


def test_orchestrator_run_mock(tmp_path: Path) -> None:
    config = GlobalConfig(
        orchestrator=OrchestratorConfig(
            max_cycles=2, work_dir=tmp_path, execution_mode=ExecutionMode.MOCK
        ),
        generator=GeneratorConfig(type=GeneratorType.MOCK),
        oracle=OracleConfig(type=OracleType.MOCK),
        trainer=TrainerConfig(type=TrainerType.MOCK),
        dynamics=DynamicsConfig(type=DynamicsType.MOCK),
        validator=ValidatorConfig(type=ValidatorType.MOCK),
    )
    orch = Orchestrator(config)
    orch.run()

    # Verify files created
    # MockTrainer creates potential.yace
    assert (tmp_path / "potential.yace").exists()

    # Verify state
    # After run, current_cycle should be max_cycles (2)? No, loop range(start, max).
    # If start is 0, it runs for 0, then 1.
    # At end of loop 0, it saves state?
    # State update cycle happens at start of loop: update_cycle(cycle+1).
    # So for cycle=0, it updates to 1.
    # For cycle=1, it updates to 2.
    # So final state should be 2.

    sm = orch.state_manager
    # Reload state from file
    sm_new = type(sm)(tmp_path)
    assert sm_new.state.current_cycle == 2
