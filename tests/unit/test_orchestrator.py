from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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

    assert (tmp_path / "potential.yace").exists()

    sm = orch.state_manager
    # Reload state from file
    sm_new = type(sm)(tmp_path)
    # 2 cycles run.
    assert sm_new.state.current_cycle == 2


def test_orchestrator_active_learning_loop(tmp_path: Path) -> None:
    """Test the Active Learning logic: Cold Start vs OTF."""

    with patch("mlip_autopipec.core.orchestrator.MockGenerator") as MockGen, \
         patch("mlip_autopipec.core.orchestrator.MockDynamics") as MockDyn, \
         patch("mlip_autopipec.core.orchestrator.MockTrainer") as MockTrain, \
         patch("mlip_autopipec.core.orchestrator.MockOracle") as MockOracleCls, \
         patch("mlip_autopipec.core.orchestrator.MockValidator") as MockVal:

        # Setup mocks behavior
        mock_gen = MockGen.return_value
        mock_dyn = MockDyn.return_value
        mock_train = MockTrain.return_value
        mock_oracle = MockOracleCls.return_value
        mock_val = MockVal.return_value

        # Generator returns iterator of structures - Must return new iterator each call
        s1 = MagicMock()
        s1.uncertainty_score = 0.1
        mock_gen.explore.side_effect = lambda *args, **kwargs: iter([s1])

        # Oracle needs to consume input to trigger lazy generation
        def consume_and_return(iterator: Iterator[Any]) -> Iterator[Any]:
            list(iterator) # Trigger generator -> dynamics
            return iter([s1])

        mock_oracle.compute.side_effect = consume_and_return

        # Trainer returns potential
        mock_pot = MagicMock()
        mock_pot.path = tmp_path / "potential.yace"
        mock_pot.path.touch()
        mock_train.train.return_value = mock_pot

        # Dynamics returns trajectory with one halted structure (high gamma)
        s_halt = MagicMock()
        s_halt.uncertainty_score = 10.0 # Above threshold 5.0
        mock_dyn.simulate.side_effect = lambda *args, **kwargs: iter([s_halt])

        # Validator returns result
        mock_val.validate.return_value = MagicMock(passed=True)

        config = GlobalConfig(
            orchestrator=OrchestratorConfig(
                max_cycles=2, work_dir=tmp_path, execution_mode=ExecutionMode.MOCK
            ),
            generator=GeneratorConfig(type=GeneratorType.MOCK),
            oracle=OracleConfig(type=OracleType.MOCK),
            trainer=TrainerConfig(type=TrainerType.MOCK),
            dynamics=DynamicsConfig(type=DynamicsType.MOCK, max_gamma_threshold=5.0),
            validator=ValidatorConfig(type=ValidatorType.MOCK),
        )

        orch = Orchestrator(config)
        orch.run()

        # Verification

        assert mock_gen.explore.call_count >= 2
        assert mock_train.train.call_count == 2

        # Verify Dynamics was called in cycle 2
        assert mock_dyn.simulate.call_count >= 1


def test_orchestrator_create_generators(tmp_path: Path) -> None:
    # Create a dummy seed file
    seed_path = tmp_path / "seed.xyz"
    seed_path.write_text("2\n\nHe 0 0 0\nHe 1 0 0")

    # Test creation of different generators
    for g_type in [GeneratorType.RANDOM, GeneratorType.M3GNET, GeneratorType.ADAPTIVE]:
        config = GlobalConfig(
            orchestrator=OrchestratorConfig(work_dir=tmp_path),
            generator=GeneratorConfig(
                type=g_type,
                seed_structure_path=seed_path
            ),
            oracle=OracleConfig(),
            trainer=TrainerConfig(),
            dynamics=DynamicsConfig(),
            validator=ValidatorConfig(),
        )

        orch = Orchestrator(config)
        assert orch.generator is not None
