from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.config.config_model import (
    Config,
    DFTConfig,
    LammpsConfig,
    OracleConfig,
    OrchestratorConfig,
    ProjectConfig,
    SelectionConfig,
    StructureGenConfig,
    TrainingConfig,
    ValidationConfig,
)
from mlip_autopipec.factory import create_components
from mlip_autopipec.orchestration.interfaces import Explorer, Oracle, Selector, Trainer, Validator
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle
from mlip_autopipec.physics.oracle.manager import DFTManager
from mlip_autopipec.physics.structure_gen.explorer import AdaptiveExplorer
from mlip_autopipec.validation.runner import ValidationRunner


@pytest.fixture
def mock_config(tmp_path: Path) -> Config:
    data_file = tmp_path / "data.xyz"
    data_file.touch()

    return Config(
        project=ProjectConfig(name="Test"),
        exploration=StructureGenConfig(strategy="adaptive"),
        selection=SelectionConfig(method="activeset"),
        oracle=OracleConfig(method="dft"),
        training=TrainingConfig(dataset_path=data_file),
        orchestrator=OrchestratorConfig(),
        validation=ValidationConfig(run_validation=True),
        lammps=LammpsConfig(),
        dft=DFTConfig(pseudopotentials={}),
    )


def test_create_components_adaptive_dft_validation(mock_config: Config) -> None:
    with (
        patch("mlip_autopipec.factory.AdaptiveExplorer") as MockAdaptive,
        patch("mlip_autopipec.factory.DFTManager") as MockDFT,
        patch("mlip_autopipec.factory.PacemakerTrainer") as MockTrainer,
        patch("mlip_autopipec.factory.ValidationRunner") as MockRunner,
        patch("mlip_autopipec.factory.ActiveSetSelector") as MockSelector,
    ):
        explorer, selector, oracle, trainer, validator = create_components(mock_config)

        assert explorer == MockAdaptive.return_value
        assert selector == MockSelector.return_value
        assert oracle == MockDFT.return_value
        assert trainer == MockTrainer.return_value
        assert validator == MockRunner.return_value


def test_create_components_mock_fallback(mock_config: Config) -> None:
    mock_config.exploration.strategy = "unknown_strategy"
    mock_config.oracle.method = "mock"
    mock_config.validation.run_validation = False

    with (
        patch("mlip_autopipec.factory.MockExplorer") as MockExpl,
        patch("mlip_autopipec.factory.MockOracle") as MockOra,
        patch("mlip_autopipec.factory.PacemakerTrainer") as MockTrainer,
        patch("mlip_autopipec.factory.ActiveSetSelector") as MockSelector,
    ):
        explorer, selector, oracle, trainer, validator = create_components(mock_config)

        assert explorer == MockExpl.return_value
        assert oracle == MockOra.return_value
        assert trainer == MockTrainer.return_value
        assert selector == MockSelector.return_value
        assert validator is None
