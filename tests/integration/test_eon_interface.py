from unittest.mock import MagicMock, patch

from mlip_autopipec.config.config_model import (
    Config,
    OracleConfig,
    OrchestratorConfig,
    ProjectConfig,
    SelectionConfig,
    StructureGenConfig,
    TrainingConfig,
    ValidationConfig,
)
from mlip_autopipec.orchestration.orchestrator import Orchestrator


@patch("mlip_autopipec.orchestration.orchestrator.StateManager")
def test_orchestrator_akmc_flow(mock_state_manager, tmp_path):
    # Setup mocks
    # Mock state manager to return no state (fresh start)
    mock_state_manager.return_value.load.return_value = None

    explorer = MagicMock()
    selector = MagicMock()
    oracle = MagicMock()
    trainer = MagicMock()
    validator = MagicMock()

    # Create dummy dataset
    dataset_path = tmp_path / "data.xyz"
    dataset_path.touch()

    # Create config with akmc strategy
    config = Config(
        project=ProjectConfig(name="test"),
        training=TrainingConfig(dataset_path=dataset_path),
        orchestrator=OrchestratorConfig(max_iterations=1),
        exploration=StructureGenConfig(strategy="akmc"),
        selection=SelectionConfig(),
        oracle=OracleConfig(),
        validation=ValidationConfig(run_validation=False)
    )

    # Mock explorer returning candidates
    explorer.explore.return_value = []
    selector.select.return_value = []
    oracle.compute.return_value = []
    trainer.train.return_value = tmp_path / "potential.yace"
    (tmp_path / "potential.yace").touch()

    orch = Orchestrator(config, explorer, selector, oracle, trainer, validator)

    # Mock production deployer
    deployer = MagicMock()
    orch.production_deployer = deployer

    # Run
    orch.run()

    explorer.explore.assert_called_once()

    # Verify production deployment was called
    deployer.deploy.assert_called()
