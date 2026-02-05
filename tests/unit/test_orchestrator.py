import pytest
from unittest.mock import MagicMock
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.domain_models.validation import ValidationResult

def test_orchestrator_loop() -> None:
    """Test that orchestrator runs the loop correctly."""
    # Setup Config
    config = GlobalConfig(max_cycles=2, execution_mode="mock")

    # Setup Mocks
    explorer = MagicMock()
    oracle = MagicMock()
    trainer = MagicMock()
    validator = MagicMock()

    # Setup Returns
    explorer.generate_candidates.return_value = [MagicMock(spec=StructureMetadata)]
    oracle.calculate.return_value = [MagicMock(spec=StructureMetadata)]
    trainer.train.return_value = "potential.yace"
    validator.validate.return_value = ValidationResult(passed=True, metrics=[])

    # Run
    orch = Orchestrator(config, explorer, oracle, trainer, validator)
    orch.run()

    # Verify calls
    assert explorer.generate_candidates.call_count == 2
    assert oracle.calculate.call_count == 2
    assert trainer.train.call_count == 2
    assert validator.validate.call_count == 2
