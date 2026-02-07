from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlip_autopipec.config.base_config import GlobalConfig
from mlip_autopipec.orchestrator.simple_orchestrator import SimpleOrchestrator


def test_initialization(temp_config: Path) -> None:
    config = GlobalConfig.from_yaml(temp_config)
    orchestrator = SimpleOrchestrator(config)
    assert orchestrator.config == config
    # Check components initialized
    assert orchestrator.oracle is not None
    assert orchestrator.trainer is not None


def test_run_logic(temp_config: Path) -> None:
    """
    Test that run executes the loop.
    We mock the components to verify orchestration logic.
    """
    config = GlobalConfig.from_yaml(temp_config)
    orchestrator = SimpleOrchestrator(config)

    # Mock components on the instance to spy on calls
    orchestrator.oracle = MagicMock()
    orchestrator.trainer = MagicMock()
    orchestrator.dynamics = MagicMock()
    orchestrator.structure_generator = MagicMock()
    orchestrator.validator = MagicMock()

    # Configure Mocks
    orchestrator.structure_generator.get_candidates.return_value = []
    # Loop needs to continue: validation fails, dynamics halts
    mock_dynamics_result = MagicMock()
    mock_dynamics_result.halted = True
    mock_dynamics_result.structures = []
    orchestrator.dynamics.run_exploration.return_value = mock_dynamics_result

    mock_validation_result = MagicMock()
    mock_validation_result.passed = False
    orchestrator.validator.validate.return_value = mock_validation_result

    # Train needs to return a Potential
    mock_potential = MagicMock()
    orchestrator.trainer.train.return_value = mock_potential

    orchestrator.run()

    # Verify calls
    assert orchestrator.structure_generator.get_candidates.called
    assert orchestrator.oracle.compute.called
    assert orchestrator.trainer.train.called
    assert orchestrator.validator.validate.called
    assert orchestrator.dynamics.run_exploration.called
    # Should run 2 iterations (from temp_config max_iterations=2)
    assert orchestrator.trainer.train.call_count == 2


def test_unimplemented_oracle(temp_config: Path) -> None:
    # Create config with valid pydantic type but unimplemented logic
    config = GlobalConfig.from_yaml(temp_config)
    config.oracle.type = "quantum_espresso"  # Allowed by Pydantic

    with pytest.raises(NotImplementedError):
        SimpleOrchestrator(config)


def test_exploration_converged(temp_config: Path) -> None:
    config = GlobalConfig.from_yaml(temp_config)
    orchestrator = SimpleOrchestrator(config)

    # Mock components
    orchestrator.oracle = MagicMock()
    orchestrator.trainer = MagicMock()
    orchestrator.dynamics = MagicMock()
    orchestrator.structure_generator = MagicMock()
    orchestrator.validator = MagicMock()

    # Configure Mocks
    orchestrator.structure_generator.get_candidates.return_value = []

    # Validation fails (so we don't break there)
    orchestrator.validator.validate.return_value.passed = False

    # Dynamics converges (halted=False)
    mock_res = MagicMock()
    mock_res.halted = False
    orchestrator.dynamics.run_exploration.return_value = mock_res

    orchestrator.run()

    # Should verify that it broke the loop (called only once)
    # even though max_iterations=2
    assert orchestrator.trainer.train.call_count == 1
