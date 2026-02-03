from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlip_autopipec.config.config_model import Config, EonConfig, ProjectConfig, TrainingConfig
from mlip_autopipec.physics.dynamics.eon_wrapper import EonWrapper
from mlip_autopipec.physics.structure_gen.explorer import AKMCExplorer


@pytest.fixture
def mock_config(tmp_path: Path) -> Config:
    dataset = tmp_path / "dataset.xyz"
    dataset.write_text("2\n\nCu 0 0 0\nCu 2 0 0")

    return Config(
        project=ProjectConfig(name="test"),
        training=TrainingConfig(dataset_path=dataset),
        eon=EonConfig(command="echo"),
    )

def test_akmc_explorer(mock_config: Config, tmp_path: Path) -> None:
    mock_wrapper = MagicMock(spec=EonWrapper)
    mock_wrapper.run_akmc.return_value = [] # Return empty list

    explorer = AKMCExplorer(mock_config, mock_wrapper)

    candidates = explorer.explore(None, tmp_path)

    assert candidates == []

    # Check if run_akmc was called with correct start structure
    mock_wrapper.run_akmc.assert_called_once()
    args = mock_wrapper.run_akmc.call_args
    # potential_path, structure_path, work_dir
    assert args[0][0] is None
    assert args[0][1].name == "start_structure.xyz"
    assert args[0][2] == tmp_path
