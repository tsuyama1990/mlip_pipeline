from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.config.config_model import (
    Config,
    EonConfig,
    OracleConfig,
    OrchestratorConfig,
    ProjectConfig,
    SelectionConfig,
    StructureGenConfig,
    TrainingConfig,
    ValidationConfig,
)
from mlip_autopipec.domain_models.exploration import ExplorationMethod, ExplorationTask
from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.orchestration.interfaces import Oracle, Selector, Trainer, Validator
from pathlib import Path
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.physics.structure_gen.explorer import AdaptiveExplorer


@pytest.fixture
def uat_config(tmp_path: Path) -> Config:
    # Create seed first because Pydantic FilePath validates existence
    dataset_path = tmp_path / "seed.xyz"
    from ase.io import write
    write(dataset_path, Atoms("Cu"))

    return Config(
        project=ProjectConfig(name="UAT_Cycle06"),
        training=TrainingConfig(dataset_path=dataset_path),
        orchestrator=OrchestratorConfig(max_iterations=1),
        exploration=StructureGenConfig(strategy="adaptive"),
        selection=SelectionConfig(method="random"),
        oracle=OracleConfig(method="mock"),
        validation=ValidationConfig(run_validation=False),
        eon=EonConfig()
    )

def test_uat_cycle06_akmc_production(uat_config: Config, tmp_path: Path) -> None:
    # This UAT verifies that Orchestrator runs AKMC and then Finalizes.

    # Mocks
    mock_selector = MagicMock(spec=Selector)
    mock_oracle = MagicMock(spec=Oracle)
    mock_trainer = MagicMock(spec=Trainer)
    mock_validator = MagicMock(spec=Validator)

    # Setup Mocks
    mock_selector.select.return_value = []
    mock_oracle.compute.return_value = []
    # Mock trainer to return a valid path for potential
    mock_trainer.update_dataset.return_value = tmp_path / "dataset.xyz"
    potential_file = tmp_path / "potential.yace"
    potential_file.touch()
    mock_trainer.train.return_value = potential_file

    # Scenario 1: AKMC is triggered.
    # We need Orchestrator to use AdaptiveExplorer.
    real_explorer = AdaptiveExplorer(uat_config)

    # Mock Policy to return AKMC
    with (
        patch("mlip_autopipec.physics.structure_gen.policy.AdaptivePolicy.decide_strategy") as mock_policy,
        patch("mlip_autopipec.physics.dynamics.eon_wrapper.EonWrapper.run_akmc", return_value=0) as mock_eon_run,
        patch("mlip_autopipec.physics.structure_gen.explorer.AdaptiveExplorer._collect_eon_results", return_value=[]),
    ):
        mock_policy.return_value = [ExplorationTask(method=ExplorationMethod.AKMC)]

        orchestrator = Orchestrator(
            config=uat_config,
            explorer=real_explorer, # Use real explorer to test AKMC wiring
            selector=mock_selector,
            oracle=mock_oracle,
            trainer=mock_trainer,
            validator=mock_validator
        )

        # Patch state manager to avoid file writes/reads of state.json
        orchestrator.state_manager = MagicMock()

        # Pre-create a potential so AKMC can run (it requires a potential)
        initial_pot = tmp_path / "initial_pot.yace"
        initial_pot.touch()

        orchestrator.state = WorkflowState(
            current_potential_path=initial_pot
        )

        # Run
        with patch("mlip_autopipec.infrastructure.production.ProductionDeployer.deploy") as mock_deploy:
            orchestrator.run()

            # Verify AKMC was run
            mock_eon_run.assert_called()

            # Verify Finalize (Production) was run
            mock_deploy.assert_called()

            # Verify manifest content passed to deploy
            call_args = mock_deploy.call_args
            assert call_args
            manifest = call_args.kwargs['manifest']
            assert manifest.version == "1.0.0"
            assert manifest.author == "UAT_Cycle06"
