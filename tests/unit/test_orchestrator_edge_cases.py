from pathlib import Path
from unittest.mock import MagicMock

from mlip_autopipec.config.base_config import GlobalConfig
from mlip_autopipec.orchestrator.simple_orchestrator import SimpleOrchestrator


def test_empty_candidates(temp_config: Path) -> None:
    """Test that the orchestrator handles empty candidate generation gracefully."""
    config = GlobalConfig.from_yaml(temp_config)
    orchestrator = SimpleOrchestrator(config)

    # Mock components
    orchestrator.structure_generator = MagicMock()
    orchestrator.oracle = MagicMock()
    orchestrator.trainer = MagicMock()
    orchestrator.dynamics = MagicMock()
    orchestrator.validator = MagicMock()

    # Return empty list
    orchestrator.structure_generator.get_candidates.return_value = []

    # Mock other methods to avoid errors if called (though loop logic might skip them)
    # If candidates empty, loop over candidates in oracle.compute does nothing.
    # Train might fail if dataset empty?
    # SimpleOrchestrator.run extends dataset with candidates.
    # If dataset empty, trainer.train called with empty list.
    # MockTrainer handles empty list fine (len=0).

    # Mock potential
    orchestrator.trainer.train.return_value = MagicMock()
    orchestrator.validator.validate.return_value.passed = True # Stop loop early

    orchestrator.run()

    # Verify candidates called
    assert orchestrator.structure_generator.get_candidates.called
    # Verify train called
    assert orchestrator.trainer.train.called


def test_validation_error_propagation(temp_config: Path) -> None:
    """Test that exceptions during validation are caught/logged and re-raised."""
    config = GlobalConfig.from_yaml(temp_config)
    orchestrator = SimpleOrchestrator(config)

    # Mock components
    orchestrator.structure_generator = MagicMock()
    orchestrator.oracle = MagicMock()
    orchestrator.trainer = MagicMock()
    orchestrator.dynamics = MagicMock()
    orchestrator.validator = MagicMock()

    orchestrator.structure_generator.get_candidates.return_value = []

    # Make validator raise exception
    orchestrator.validator.validate.side_effect = RuntimeError("Validator Crashed")

    # The run loop has try/except that re-raises
    import pytest
    with pytest.raises(RuntimeError, match="Validator Crashed"):
        orchestrator.run()
