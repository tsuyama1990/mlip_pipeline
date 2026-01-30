import pytest
from unittest.mock import patch
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner
from mlip_autopipec.domain_models.training import TrainingConfig, TrainingResult
from mlip_autopipec.domain_models.config import PotentialConfig


@pytest.fixture
def training_config():
    return TrainingConfig(
        max_epochs=10,
        batch_size=5,
        kappa=0.5
    )

@pytest.fixture
def potential_config():
    return PotentialConfig(
        elements=["Si", "O"],
        cutoff=4.5,
        seed=123
    )

@patch("subprocess.run")
@patch("pathlib.Path.read_text")
@patch("pathlib.Path.exists")
@patch("mlip_autopipec.infrastructure.io.dump_yaml")
def test_train_success(mock_dump_yaml, mock_exists, mock_read_text, mock_run, training_config, potential_config, tmp_path):
    # Mock existence of potential file
    mock_exists.return_value = True

    # Mock log output for RMSE parsing
    mock_read_text.return_value = """
    ...
    INFO: Final RMSE Energy: 1.5 meV/atom
    INFO: Final RMSE Force: 0.02 eV/A
    ...
    """

    runner = PacemakerRunner(training_config, potential_config, work_dir=tmp_path)

    # Dummy dataset path
    dataset_path = tmp_path / "train.pckl.gzip"

    result = runner.train(dataset_path)

    assert isinstance(result, TrainingResult)
    assert result.status == "COMPLETED"
    assert result.validation_metrics["rmse_energy"] == 1.5
    assert result.validation_metrics["rmse_force"] == 0.02

    # Verify dump_yaml call args
    mock_dump_yaml.assert_called()
    call_args = mock_dump_yaml.call_args
    # call_args[0] is positional args (config_dict, output_path)
    config_dict = call_args[0][0]

    assert config_dict["cutoff"] == 4.5
    assert config_dict["bonds"]["element"] == ["Si", "O"]
    assert config_dict["fit"]["loss"]["kappa"] == 0.5
    assert config_dict["backend"]["batch_size"] == 5

    # Check if subprocess was called with pace_train
    mock_run.assert_called()
    args = mock_run.call_args[0][0]
    assert "pace_train" in args or args[0].endswith("pace_train")

@patch("subprocess.run")
def test_select_active_set(mock_run, training_config, potential_config, tmp_path):
    runner = PacemakerRunner(training_config, potential_config, work_dir=tmp_path)
    full_dataset = tmp_path / "full.pckl.gzip"

    # Mock output
    mock_run.return_value.returncode = 0

    pruned_dataset = runner.select_active_set(full_dataset)

    assert pruned_dataset == tmp_path / "train_active.pckl.gzip"
    mock_run.assert_called()
    args = mock_run.call_args[0][0]
    assert "pace_activeset" in args or args[0].endswith("pace_activeset")
