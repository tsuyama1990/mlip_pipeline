from pathlib import Path
from unittest.mock import patch

import pytest

from mlip_autopipec.config.config_model import (
    Config,
    OracleConfig,
    ProjectConfig,
    SelectionConfig,
    StructureGenConfig,
    TrainingConfig,
    ValidationConfig,
)
from mlip_autopipec.factory import create_components


@pytest.fixture
def mock_config(tmp_path: Path) -> Config:
    data_file = tmp_path / "data.xyz"
    data_file.touch()
    return Config(
        project=ProjectConfig(name="Test"),
        exploration=StructureGenConfig(strategy="adaptive"),
        selection=SelectionConfig(method="activeset"),
        oracle=OracleConfig(method="mock"),
        training=TrainingConfig(dataset_path=data_file, max_epochs=1),
        validation=ValidationConfig(run_validation=True),
    )


def test_create_components_validation_enabled(mock_config: Config) -> None:
    with (
        patch("mlip_autopipec.factory.AdaptiveExplorer") as MockExpl,
        patch("mlip_autopipec.factory.ActiveSetSelector") as MockSel,
        patch("mlip_autopipec.factory.MockOracle") as MockOra,
        patch("mlip_autopipec.factory.PacemakerTrainer") as MockTrain,
        patch("mlip_autopipec.factory.ValidationRunner") as MockVal,
    ):
        explorer, selector, oracle, trainer, validator = create_components(mock_config)

        assert explorer == MockExpl.return_value
        assert selector == MockSel.return_value
        assert oracle == MockOra.return_value
        assert trainer == MockTrain.return_value
        assert validator == MockVal.return_value

        MockVal.assert_called_once_with(mock_config.validation)


def test_create_components_validation_disabled(mock_config: Config) -> None:
    mock_config.validation.run_validation = False
    with (
        patch("mlip_autopipec.factory.AdaptiveExplorer"),
        patch("mlip_autopipec.factory.ActiveSetSelector"),
        patch("mlip_autopipec.factory.MockOracle"),
        patch("mlip_autopipec.factory.PacemakerTrainer"),
        patch("mlip_autopipec.factory.ValidationRunner") as MockVal,
    ):
        _, _, _, _, validator = create_components(mock_config)

        assert validator is None
        MockVal.assert_not_called()


def test_create_components_fallback_explorer(mock_config: Config) -> None:
    mock_config.exploration.strategy = "unknown"
    with (
        patch("mlip_autopipec.factory.MockExplorer") as MockExpl,
        patch("mlip_autopipec.factory.AdaptiveExplorer") as MockAdaptive,
        patch("mlip_autopipec.factory.ActiveSetSelector"),
        patch("mlip_autopipec.factory.MockOracle"),
        patch("mlip_autopipec.factory.PacemakerTrainer"),
    ):
        explorer, _, _, _, _ = create_components(mock_config)

        assert explorer == MockExpl.return_value
        MockAdaptive.assert_not_called()
