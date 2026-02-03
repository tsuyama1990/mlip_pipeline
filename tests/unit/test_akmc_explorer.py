from unittest.mock import MagicMock, patch

from mlip_autopipec.config.config_model import (
    Config,
    OracleConfig,
    ProjectConfig,
    SelectionConfig,
    StructureGenConfig,
    TrainingConfig,
    ValidationConfig,
)
from mlip_autopipec.physics.structure_gen.explorer import AKMCExplorer


@patch("mlip_autopipec.physics.structure_gen.explorer.read")
@patch("mlip_autopipec.physics.structure_gen.explorer.write")
def test_akmc_explorer(mock_write, mock_read, tmp_path):
    # Prepare data
    (tmp_path/"data.xyz").touch()

    # Minimal config
    config = Config(
        project=ProjectConfig(name="test"),
        training=TrainingConfig(dataset_path=tmp_path/"data.xyz"),
        exploration=StructureGenConfig(strategy="akmc"),
        selection=SelectionConfig(),
        oracle=OracleConfig(),
        validation=ValidationConfig(run_validation=False)
    )
    eon_wrapper = MagicMock()

    explorer = AKMCExplorer(config, eon_wrapper)

    # Mock read seed
    mock_read.return_value = [MagicMock()]

    # Mock wrapper result
    eon_wrapper.run_akmc.return_value = [MagicMock()]

    cands = explorer.explore(tmp_path/"pot.yace", tmp_path)

    assert len(cands) == 1
    assert cands[0].metadata.generation_method == "akmc"
