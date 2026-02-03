import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.io import write

from mlip_autopipec.config import (
    Config,
    OracleConfig,
    ProjectConfig,
    StructureGenConfig,
    TrainingConfig,
    ValidationConfig,
)
from mlip_autopipec.domain_models.exploration import ExplorationMethod, ExplorationTask
from mlip_autopipec.physics.structure_gen.explorer import AdaptiveExplorer


@pytest.fixture
def mock_config(temp_dir: Path) -> Config:
    (temp_dir / "data.pckl").touch()
    return Config(
        project=ProjectConfig(name="Test"),
        training=TrainingConfig(dataset_path=temp_dir / "data.pckl"),
        exploration=StructureGenConfig(strategy="adaptive"),
        oracle=OracleConfig(),
        validation=ValidationConfig(),
    )


def test_explorer_no_seed_file(mock_config: Config, temp_dir: Path) -> None:
    # Remove seed file
    mock_config.training.dataset_path.unlink()

    explorer = AdaptiveExplorer(mock_config)
    candidates = explorer.explore(potential_path=None, work_dir=temp_dir)
    assert candidates == []


def test_explorer_execution(mock_config: Config, temp_dir: Path) -> None:
    # Create seed file (XYZ) because ASE read needs valid format
    seed_path = mock_config.training.dataset_path.with_suffix(".xyz")
    mock_config.training.dataset_path = seed_path

    atoms = Atoms("Cu", cell=[3.6, 3.6, 3.6], pbc=True)
    write(seed_path, atoms)

    explorer = AdaptiveExplorer(mock_config)

    # Mock Policy to return deterministic tasks
    with patch.object(explorer, "policy") as mock_policy:
        mock_policy.decide_strategy.return_value = [
            ExplorationTask(
                method=ExplorationMethod.STATIC,
                modifiers=["strain"],
                parameters={"strain_range": 0.01},
            ),
            ExplorationTask(
                method=ExplorationMethod.STATIC,
                modifiers=["defect"],
                parameters={"defect_type": "vacancy"},
            ),
        ]

        candidates = explorer.explore(potential_path=None, work_dir=temp_dir)

        # 20 strain + 1 defect = 21 candidates
        assert len(candidates) == 21

        # Verify files created
        assert (temp_dir / "candidate_t0_0.xyz").exists()
        assert (temp_dir / "candidate_t1_0.xyz").exists()


def test_explorer_unknown_modifier(mock_config: Config, temp_dir: Path) -> None:
    seed_path = mock_config.training.dataset_path.with_suffix(".xyz")
    mock_config.training.dataset_path = seed_path
    atoms = Atoms("Cu", cell=[3.6, 3.6, 3.6], pbc=True)
    write(seed_path, atoms)

    explorer = AdaptiveExplorer(mock_config)

    with patch.object(explorer, "policy") as mock_policy:
        mock_policy.decide_strategy.return_value = [
            ExplorationTask(method=ExplorationMethod.STATIC, modifiers=["unknown"])
        ]

        candidates = explorer.explore(potential_path=None, work_dir=temp_dir)
        assert len(candidates) == 0
