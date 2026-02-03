from pathlib import Path
from unittest.mock import patch

import pytest

from mlip_autopipec.config.config_model import (
    Config,
    DFTConfig,
    OracleConfig,
    OrchestratorConfig,
    ProjectConfig,
    SelectionConfig,
    StructureGenConfig,
    TrainingConfig,
    ValidationConfig,
)
from mlip_autopipec.main import create_components
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle


@pytest.fixture
def base_config(tmp_path: Path) -> Config:
    # Create a dummy dataset file because TrainingConfig requires FilePath
    dset = tmp_path / "data.pckl"
    dset.touch()

    return Config(
        project=ProjectConfig(name="Test"),
        training=TrainingConfig(dataset_path=dset),
        orchestrator=OrchestratorConfig(max_iterations=1),
        exploration=StructureGenConfig(strategy="adaptive"),
        selection=SelectionConfig(),
        oracle=OracleConfig(method="dft"),
        validation=ValidationConfig(run_validation=True),
        dft=DFTConfig(pseudopotentials={"Si": "Si.upf"}),
    )


def test_create_components_full(base_config: Config) -> None:
    with (
        patch("mlip_autopipec.main.LammpsRunner") as MockLammps,
        patch("mlip_autopipec.main.OTFLoop") as MockOTF,
        patch("mlip_autopipec.main.AdaptiveExplorer") as MockAdaptExp,
        patch("mlip_autopipec.main.DFTManager") as MockDFT,
        patch("mlip_autopipec.main.ValidationRunner") as MockValRunner,
    ):
        explorer, selector, oracle, trainer, validator = create_components(base_config)

        # Verify calls
        MockDFT.assert_called_once()
        MockValRunner.assert_called_once()
        MockAdaptExp.assert_called_once()

        # We didn't provide lammps config, so OTF should be None (or not created)
        MockOTF.assert_not_called()
        MockLammps.assert_not_called()


def test_create_components_mock_fallback(base_config: Config) -> None:
    # Test fallback to MockExplorer for unknown strategy
    base_config.exploration.strategy = "unknown_strategy"
    # Test MockOracle
    base_config.oracle.method = "mock"
    # Test No Validation
    base_config.validation.run_validation = False

    with (
         patch("mlip_autopipec.main.AdaptiveExplorer") as MockAdaptExp,
         patch("mlip_autopipec.main.DFTManager") as MockDFT,
         patch("mlip_autopipec.main.ValidationRunner") as MockValRunner,
    ):
        explorer, selector, oracle, trainer, validator = create_components(base_config)

        assert isinstance(explorer, MockExplorer)
        assert isinstance(oracle, MockOracle)
        assert validator is None

        MockAdaptExp.assert_not_called()
        MockDFT.assert_not_called()
        MockValRunner.assert_not_called()


def test_create_components_lammps(base_config: Config) -> None:
    from mlip_autopipec.config.config_model import LammpsConfig
    base_config.lammps = LammpsConfig()

    with (
        patch("mlip_autopipec.main.LammpsRunner") as MockLammps,
        patch("mlip_autopipec.main.OTFLoop") as MockOTF,
        patch("mlip_autopipec.main.AdaptiveExplorer") as MockAdaptExp,
        patch("mlip_autopipec.main.DFTManager"),
        patch("mlip_autopipec.main.ValidationRunner"),
    ):
        create_components(base_config)

        MockLammps.assert_called_once()
        MockOTF.assert_called_once()
        # Verify OTF injected into Explorer
        MockAdaptExp.assert_called_once()
        _, kwargs = MockAdaptExp.call_args
        assert "otf_loop" in kwargs
