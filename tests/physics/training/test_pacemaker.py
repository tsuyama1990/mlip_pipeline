import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner
from mlip_autopipec.domain_models.training import TrainingConfig
from mlip_autopipec.domain_models.config import PotentialConfig

@pytest.fixture
def training_config():
    return TrainingConfig(batch_size=32, max_epochs=10)

@pytest.fixture
def potential_config():
    return PotentialConfig(elements=["Ti", "O"], cutoff=5.0)

def test_train_success(tmp_path, training_config, potential_config):
    runner = PacemakerRunner(work_dir=tmp_path)
    dataset_path = tmp_path / "data.pckl.gzip"
    dataset_path.touch()

    # Mock subprocess and log file reading
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        # We need to simulate the creation of artifacts by the subprocess
        def side_effect(*args, **kwargs):
             (tmp_path / "output_potential.yace").touch()
             (tmp_path / "log.txt").write_text("... Final RMSE Energy: 1.23 meV ... Final RMSE Force: 0.04 eV/A ...")
             return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        result = runner.train(
            dataset_path=dataset_path,
            training_config=training_config,
            potential_config=potential_config
        )

        assert result.potential_path.exists()
        assert result.validation_metrics["rmse_energy"] == 1.23
        assert result.validation_metrics["rmse_force"] == 0.04

def test_train_initial_potential(tmp_path, training_config, potential_config):
    training_config.initial_potential = Path("initial.yace")
    runner = PacemakerRunner(work_dir=tmp_path)
    dataset_path = tmp_path / "data.pckl.gzip"
    dataset_path.touch()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        (tmp_path / "output_potential.yace").touch()
        runner.train(dataset_path, training_config, potential_config)

        # Check input.yaml content
        import yaml
        with (tmp_path / "input.yaml").open() as f:
             data = yaml.safe_load(f)
             assert data["potential"]["initial_potential"] == str(Path("initial.yace").absolute())

def test_activeset_selection(tmp_path):
    runner = PacemakerRunner(work_dir=tmp_path)
    dataset_path = tmp_path / "data.pckl.gzip"
    dataset_path.touch()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        def side_effect(*args, **kwargs):
            (tmp_path / "data_active.pckl.gzip").touch()
            return MagicMock(returncode=0)
        mock_run.side_effect = side_effect

        new_path = runner.select_active_set(dataset_path)

        assert new_path == tmp_path / "data_active.pckl.gzip"
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "pace_activeset"

def test_train_failure(tmp_path, training_config, potential_config):
    runner = PacemakerRunner(work_dir=tmp_path)
    dataset_path = tmp_path / "data.pckl.gzip"
    dataset_path.touch()

    import subprocess
    with patch("subprocess.run") as mock_run:
        def side_effect(*args, **kwargs):
             (tmp_path / "log.txt").write_text("Crash log")
             raise subprocess.CalledProcessError(1, "pace_train")
        mock_run.side_effect = side_effect

        result = runner.train(dataset_path, training_config, potential_config)
        assert result.status.value == "FAILED"
        assert "Crash log" in result.log_content

def test_train_executable_missing(tmp_path, training_config, potential_config):
    runner = PacemakerRunner(work_dir=tmp_path)
    dataset_path = tmp_path / "data.pckl.gzip"
    dataset_path.touch()

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("pace_train")
        result = runner.train(dataset_path, training_config, potential_config)
        assert result.status.value == "FAILED"
        assert "pace_train executable not found" in result.log_content

def test_activeset_failure(tmp_path):
    runner = PacemakerRunner(work_dir=tmp_path)
    dataset_path = tmp_path / "data.pckl.gzip"
    dataset_path.touch()

    import subprocess
    with patch("subprocess.run") as mock_run:
         mock_run.side_effect = subprocess.CalledProcessError(1, "pace_activeset")

         with pytest.raises(RuntimeError, match="Active set selection failed"):
             runner.select_active_set(dataset_path)

def test_activeset_no_output(tmp_path):
    runner = PacemakerRunner(work_dir=tmp_path)
    dataset_path = tmp_path / "data.pckl.gzip"
    dataset_path.touch()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        # No side effect to create file

        with pytest.raises(RuntimeError, match="Active set output file not found"):
             runner.select_active_set(dataset_path)

def test_parse_log_missing(tmp_path):
    runner = PacemakerRunner(work_dir=tmp_path)
    metrics = runner._parse_log(tmp_path / "nonexistent.log")
    assert metrics == {}
