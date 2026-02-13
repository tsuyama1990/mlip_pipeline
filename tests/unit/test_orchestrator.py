from pathlib import Path
from unittest.mock import MagicMock, call

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

    sm = orch.state_manager
    # Reload state from file
    sm_new = type(sm)(tmp_path)
    # 2 cycles run.
    assert sm_new.state.current_cycle == 2
    # Ensure potential path is updated in state
    assert sm_new.state.active_potential_path is not None
    assert sm_new.state.active_potential_path.exists()
    assert sm_new.state.active_potential_path.suffix == ".yace"


def test_orchestrator_active_learning_loop(tmp_path: Path) -> None:
    """Test the Active Learning logic using real Mocks (integration style)."""

    config = GlobalConfig(
        orchestrator=OrchestratorConfig(
            max_cycles=2, work_dir=tmp_path, execution_mode=ExecutionMode.MOCK
        ),
        generator=GeneratorConfig(type=GeneratorType.MOCK),
        oracle=OracleConfig(type=OracleType.MOCK),
        trainer=TrainerConfig(type=TrainerType.MOCK),
        # Configure dynamics to halt: threshold 5.0, mock dynamics yields score > 5.0 around frame 4
        dynamics=DynamicsConfig(type=DynamicsType.MOCK, max_gamma_threshold=5.0),
        validator=ValidatorConfig(type=ValidatorType.MOCK),
    )

    orch = Orchestrator(config)

    # Spy on components to verify interactions without replacing logic
    # We wrap the bound methods with MagicMock(wraps=...)
    orch.generator.generate_local_candidates = MagicMock(  # type: ignore[method-assign]
        wraps=orch.generator.generate_local_candidates
    )
    orch.dynamics.simulate = MagicMock(wraps=orch.dynamics.simulate)  # type: ignore[method-assign]

    orch.run()

    # Verification

    # Cycle 1 (Cold Start) + Cycle 2 (OTF)
    # Dynamics should be called in Cycle 2 (when potential exists)
    assert orch.dynamics.simulate.called

    # OTF Loop Verification
    # With max_gamma_threshold=5.0, MockDynamics should eventually yield a frame with score > 5.0
    # triggering generate_local_candidates.
    assert orch.generator.generate_local_candidates.called


def test_orchestrator_create_generators(tmp_path: Path) -> None:
    # Create a dummy seed file
    seed_path = tmp_path / "seed.xyz"
    seed_path.write_text("2\n\nHe 0 0 0\nHe 1 0 0")

    # Test creation of different generators
    for g_type in [GeneratorType.RANDOM, GeneratorType.M3GNET, GeneratorType.ADAPTIVE]:
        config = GlobalConfig(
            orchestrator=OrchestratorConfig(work_dir=tmp_path),
            generator=GeneratorConfig(type=g_type, seed_structure_path=seed_path),
            oracle=OracleConfig(),
            trainer=TrainerConfig(),
            dynamics=DynamicsConfig(),
            validator=ValidatorConfig(),
        )

        orch = Orchestrator(config)
        assert orch.generator is not None


def test_orchestrator_cold_start_mock(tmp_path: Path) -> None:
    """Test Orchestrator runs correctly when no potential exists (Cycle 0)."""
    # Configure so that state has no potential path initially
    config = GlobalConfig(
        orchestrator=OrchestratorConfig(
            max_cycles=1, work_dir=tmp_path, execution_mode=ExecutionMode.MOCK
        ),
        generator=GeneratorConfig(type=GeneratorType.MOCK),
        oracle=OracleConfig(type=OracleType.MOCK),
        trainer=TrainerConfig(type=TrainerType.MOCK),
        dynamics=DynamicsConfig(type=DynamicsType.MOCK),
        validator=ValidatorConfig(type=ValidatorType.MOCK),
    )
    orch = Orchestrator(config)

    # Ensure no initial potential
    assert orch.state_manager.state.active_potential_path is None

    orch.run()

    # After run, potential should be created
    assert orch.state_manager.state.active_potential_path is not None
    assert orch.state_manager.state.active_potential_path.exists()


def test_orchestrator_state_saving(tmp_path: Path) -> None:
    """Test that state manager saves are called during the loop."""
    config = GlobalConfig(
        orchestrator=OrchestratorConfig(
            max_cycles=1, work_dir=tmp_path, execution_mode=ExecutionMode.MOCK
        ),
        generator=GeneratorConfig(type=GeneratorType.MOCK),
        oracle=OracleConfig(type=OracleType.MOCK),
        trainer=TrainerConfig(type=TrainerType.MOCK),
        dynamics=DynamicsConfig(type=DynamicsType.MOCK),
        validator=ValidatorConfig(type=ValidatorType.MOCK),
    )
    orch = Orchestrator(config)

    # Mock state manager save
    orch.state_manager.save = MagicMock() # type: ignore[method-assign]

    orch.run()

    # Verify save calls
    # Should be called at start of cycle, after exploration, after oracle, after training, after validation/end
    # At least 5 times per cycle
    assert orch.state_manager.save.call_count >= 5
