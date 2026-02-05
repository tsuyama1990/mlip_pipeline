from unittest.mock import MagicMock

from ase import Atoms
from mlip_autopipec.orchestration.orchestrator import Orchestrator

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.domain_models.validation import ValidationResult


def test_orchestrator_loop() -> None:
    # Setup Mocks
    mock_explorer = MagicMock()
    mock_oracle = MagicMock()
    mock_trainer = MagicMock()
    mock_validator = MagicMock()

    # Create dummy structure
    dummy_structure = StructureMetadata(
        id="s1",
        source="mock",
        generation_method="random",
        structure=Atoms("H2"),
    )

    # Configure Mocks
    mock_explorer.generate_candidates.return_value = [dummy_structure]
    mock_oracle.calculate.return_value = [dummy_structure]
    mock_trainer.train.return_value = "potential.yace"
    mock_validator.validate.return_value = ValidationResult(passed=True, metrics=[])

    # Config
    config = GlobalConfig(max_cycles=2)

    # Instantiate Orchestrator
    orchestrator = Orchestrator(
        explorer=mock_explorer,
        oracle=mock_oracle,
        trainer=mock_trainer,
        validator=mock_validator,
        config=config,
    )

    # Run
    orchestrator.run()

    # Verify Interactions
    assert mock_explorer.generate_candidates.call_count == 2
    assert mock_oracle.calculate.call_count == 2
    assert mock_trainer.train.call_count == 2
    assert mock_validator.validate.call_count == 2

    # Verify call arguments for the first cycle
    mock_explorer.generate_candidates.assert_any_call(config.exploration)
    mock_oracle.calculate.assert_any_call([dummy_structure], config.dft)
