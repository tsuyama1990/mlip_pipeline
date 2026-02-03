from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.config.config_model import (
    Config,
    DFTConfig,
    StructureGenConfig,
    LammpsConfig,
    OracleConfig,
    ProjectConfig,
    SelectionConfig,
    TrainingConfig,
    ValidationConfig,
    OrchestratorConfig,
)
from mlip_autopipec.main import create_components
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle
from mlip_autopipec.physics.oracle.manager import DFTManager
from mlip_autopipec.physics.selection.selector import ActiveSetSelector
from mlip_autopipec.physics.structure_gen.explorer import AdaptiveExplorer
from mlip_autopipec.physics.training.pacemaker import PacemakerTrainer
from mlip_autopipec.validation.runner import ValidationRunner


@pytest.fixture
def base_config(tmp_path: Path) -> Config:
    # Ensure dataset path exists for FilePath validation
    data_path = tmp_path / "data.xyz"
    data_path.touch()

    return Config(
        project=ProjectConfig(name="Test"),
        training=TrainingConfig(dataset_path=data_path),
        orchestrator=OrchestratorConfig(max_iterations=1),
        exploration=StructureGenConfig(strategy="adaptive"),
        selection=SelectionConfig(method="random"),
        oracle=OracleConfig(method="mock"),
        validation=ValidationConfig(run_validation=False),
        dft=None,
        lammps=None,
    )


def test_create_components_basic(base_config: Config) -> None:
    # Basic setup with Mock Oracle and no validation
    explorer, selector, oracle, trainer, validator = create_components(base_config)

    assert isinstance(explorer, AdaptiveExplorer)
    assert isinstance(selector, ActiveSetSelector)
    assert isinstance(oracle, MockOracle)
    assert isinstance(trainer, PacemakerTrainer)
    assert validator is None


def test_create_components_with_validation(base_config: Config) -> None:
    base_config.validation.run_validation = True

    explorer, selector, oracle, trainer, validator = create_components(base_config)

    assert isinstance(validator, ValidationRunner)


def test_create_components_with_dft_oracle(base_config: Config) -> None:
    base_config.oracle.method = "dft"
    base_config.dft = DFTConfig(
        pseudopotentials={"Si": "Si.upf"},
        kspacing=0.04
    )

    explorer, selector, oracle, trainer, validator = create_components(base_config)

    assert isinstance(oracle, DFTManager)


def test_create_components_dft_missing_config(base_config: Config) -> None:
    base_config.oracle.method = "dft"
    base_config.dft = None

    with pytest.raises(ValueError, match="DFT configuration missing"):
        create_components(base_config)


def test_create_components_unknown_strategy(base_config: Config) -> None:
    base_config.exploration.strategy = "unknown"

    explorer, selector, oracle, trainer, validator = create_components(base_config)

    assert isinstance(explorer, MockExplorer)


def test_create_components_with_lammps(base_config: Config) -> None:
    base_config.lammps = LammpsConfig(command="lmp")

    with (
        patch("mlip_autopipec.main.LammpsRunner") as MockRunner,
        patch("mlip_autopipec.main.OTFLoop") as MockOTF,
    ):
        create_components(base_config)

        MockRunner.assert_called_once()
        MockOTF.assert_called_once()
