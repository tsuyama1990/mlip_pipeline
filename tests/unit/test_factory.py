from pathlib import Path

import pytest

from mlip_autopipec.config import (
    Config,
    OracleConfig,
    ProjectConfig,
    SelectionConfig,
    StructureGenConfig,
    TrainingConfig,
    ValidationConfig,
)
from mlip_autopipec.factory import create_components
from mlip_autopipec.orchestration.mocks import MockExplorer, MockValidator
from mlip_autopipec.physics.structure_gen.explorer import AdaptiveExplorer


@pytest.fixture
def base_config(tmp_path: Path) -> Config:
    data_file = tmp_path / "data.pckl"
    data_file.touch()
    return Config(
        project=ProjectConfig(name="Test"),
        training=TrainingConfig(dataset_path=data_file),
        exploration=StructureGenConfig(strategy="adaptive"),
        selection=SelectionConfig(),
        oracle=OracleConfig(method="mock"),
        validation=ValidationConfig(),
    )


def test_create_components_adaptive_explorer(base_config: Config) -> None:
    base_config.exploration.strategy = "adaptive"
    exp, sel, ora, train, val = create_components(base_config)
    assert isinstance(exp, AdaptiveExplorer)


def test_create_components_mock_explorer(base_config: Config) -> None:
    base_config.exploration.strategy = "unknown_strategy"
    exp, sel, ora, train, val = create_components(base_config)
    assert isinstance(exp, MockExplorer)


def test_create_components_dft_oracle(base_config: Config) -> None:
    base_config.oracle.method = "dft"
    # Need to provide dft config or it raises ValueError
    # But Config model might allow dft=None. create_components checks it.
    # We expect ValueError if dft is None
    with pytest.raises(ValueError, match="DFT configuration missing"):
        create_components(base_config)


def test_create_components_validator(base_config: Config) -> None:
    base_config.validation.run_validation = True
    exp, sel, ora, train, val = create_components(base_config)
    assert isinstance(val, MockValidator)

    base_config.validation.run_validation = False
    exp, sel, ora, train, val = create_components(base_config)
    assert val is None
