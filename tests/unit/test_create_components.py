from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.main import create_components
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle


@pytest.fixture
def mock_config() -> MagicMock:
    config = MagicMock()
    # Remove spec=Config to avoid attribute errors if Config definition is complex or not fully loaded

    # Setup nested attributes
    config.lammps = None

    config.exploration = MagicMock()
    config.exploration.strategy = "mock"

    config.selection = MagicMock()
    config.selection.method = "random"

    config.oracle = MagicMock()
    config.oracle.method = "mock"

    config.training = MagicMock()

    config.validation = MagicMock()
    config.validation.run_validation = False

    config.dft = None

    return config


def test_create_components_mock_defaults(mock_config: MagicMock) -> None:
    explorer, selector, oracle, trainer, validator = create_components(mock_config)

    assert isinstance(explorer, MockExplorer)
    assert isinstance(oracle, MockOracle)
    assert validator is None


def test_create_components_validation_enabled(mock_config: MagicMock) -> None:
    mock_config.validation.run_validation = True

    with patch("mlip_autopipec.main.ValidationRunner") as MockRunner:
        explorer, selector, oracle, trainer, validator = create_components(mock_config)

        MockRunner.assert_called_once()
        assert validator == MockRunner.return_value


def test_create_components_unknown_strategy(mock_config: MagicMock) -> None:
    mock_config.exploration.strategy = "unknown_strategy"

    with pytest.raises(ValueError, match="Unknown exploration strategy"):
        create_components(mock_config)


def test_create_components_adaptive_no_lammps(mock_config: MagicMock) -> None:
    mock_config.exploration.strategy = "adaptive"
    # AdaptiveExplorer requires config
    # We mock AdaptiveExplorer to avoid complex init
    with patch("mlip_autopipec.main.AdaptiveExplorer") as MockAdaptive:
        create_components(mock_config)
        MockAdaptive.assert_called()


def test_create_components_dft_missing(mock_config: MagicMock) -> None:
    mock_config.oracle.method = "dft"
    mock_config.dft = None

    with pytest.raises(ValueError, match="DFT configuration missing"):
        create_components(mock_config)
