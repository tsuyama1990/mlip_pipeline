from pathlib import Path

from mlip_autopipec.config import (
    Config,
    DFTConfig,
    OracleConfig,
    ProjectConfig,
    StructureGenConfig,
    TrainingConfig,
    ValidationConfig,
)
from mlip_autopipec.factory import create_components
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockValidator
from mlip_autopipec.physics.oracle.manager import DFTManager
from mlip_autopipec.physics.structure_gen.explorer import AdaptiveExplorer
from mlip_autopipec.physics.training.pacemaker import PacemakerTrainer


def test_create_components_adaptive_dft(temp_dir: Path) -> None:
    data_file = temp_dir / "data.pckl"
    data_file.touch()

    config = Config(
        project=ProjectConfig(name="Test"),
        training=TrainingConfig(dataset_path=data_file),
        exploration=StructureGenConfig(strategy="adaptive"),
        oracle=OracleConfig(method="dft"),
        dft=DFTConfig(pseudopotentials={}),
        validation=ValidationConfig(run_validation=True),
    )

    # We need to ensure we can instantiate these without errors (e.g. imports)
    explorer, oracle, trainer, validator = create_components(config)

    assert isinstance(explorer, AdaptiveExplorer)
    assert isinstance(oracle, DFTManager)
    assert isinstance(trainer, PacemakerTrainer)
    # The factory instantiates MockValidator for validation=True currently
    assert isinstance(validator, MockValidator)


def test_create_components_mock(temp_dir: Path) -> None:
    data_file = temp_dir / "data.pckl"
    data_file.touch()

    config = Config(
        project=ProjectConfig(name="Test"),
        training=TrainingConfig(dataset_path=data_file),
        exploration=StructureGenConfig(strategy="unknown"),
        oracle=OracleConfig(method="mock"),
        validation=ValidationConfig(run_validation=False),
    )

    explorer, oracle, trainer, validator = create_components(config)

    assert isinstance(explorer, MockExplorer)
    assert isinstance(oracle, MockOracle)
    assert isinstance(trainer, PacemakerTrainer)
    assert validator is None
