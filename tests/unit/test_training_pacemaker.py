import subprocess
from unittest.mock import patch

import pytest
import yaml

from mlip_autopipec.config.schemas.training import TrainingConfig, TrainingMetrics
from mlip_autopipec.training.pacemaker import PacemakerWrapper


@pytest.fixture
def training_config():
    return TrainingConfig(
        cutoff=4.5,
        b_basis_size=200,
        kappa=0.6,
        kappa_f=100.0,
        max_iter=10
    )

def test_generate_config(training_config, tmp_path):
    """Test generating Pacemaker input YAML."""
    wrapper = PacemakerWrapper(config=training_config, work_dir=tmp_path)

    config_path = wrapper.generate_config()

    assert config_path.exists()
    assert config_path.name == "input.yaml"

    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Check mappings
    assert data['cutoff'] == 4.5
    assert data['data']['filename'] == "data/train.xyz"
    assert data['data']['test_filename'] == "data/test.xyz"
    # Check basic structure for fitting (simplification of Pacemaker config)
    # The actual structure depends on Pacemaker version, but we assume a standard mapping
    assert data['fit']['loss']['kappa'] == 0.6
    assert data['fit']['loss']['kappa_f'] == 100.0

@patch("shutil.which")
@patch("subprocess.run")
@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper.check_output")
@patch("mlip_autopipec.training.metrics.LogParser.parse_file")
def test_train_success(mock_parse, mock_check, mock_run, mock_which, training_config, tmp_path):
    """Test successful training run."""
    wrapper = PacemakerWrapper(config=training_config, work_dir=tmp_path)

    # Mock executable existence
    mock_which.return_value = "/usr/bin/pacemaker"

    # Mock subprocess success
    mock_run.return_value.returncode = 0

    # Mock output check success
    mock_check.return_value = True

    # Mock metrics parsing
    expected_metrics = TrainingMetrics(epoch=10, rmse_energy=1.5, rmse_force=0.02)
    mock_parse.return_value = expected_metrics

    result = wrapper.train()

    assert result.success is True
    assert result.metrics == expected_metrics

    # Verify subprocess call
    args, kwargs = mock_run.call_args
    # args[0] should contain 'pacemaker' and 'input.yaml'
    assert "pacemaker" in args[0]
    # The second arg is usually the config path (relative if cwd is set)
    assert "input.yaml" in args[0]

@patch("subprocess.run")
def test_train_failure_return_code(mock_run, training_config, tmp_path):
    """Test training failure (non-zero return code)."""
    wrapper = PacemakerWrapper(config=training_config, work_dir=tmp_path)

    mock_run.return_value.returncode = 1

    result = wrapper.train()

    assert result.success is False
    assert result.metrics is None

@patch("subprocess.run")
def test_train_failure_os_error(mock_run, training_config, tmp_path):
    """Test training failure due to OS error (permission denied, etc)."""
    wrapper = PacemakerWrapper(config=training_config, work_dir=tmp_path)

    mock_run.side_effect = OSError("Permission denied")

    result = wrapper.train()

    assert result.success is False

@patch("subprocess.run")
def test_train_failure_missing_executable(mock_run, training_config, tmp_path):
    """Test training failure when binary is missing."""
    wrapper = PacemakerWrapper(config=training_config, work_dir=tmp_path)

    # Actually FileNotFoundError for binary usually happens before subprocess if checking which,
    # but subprocess.run raises FileNotFoundError if executable not found in path
    mock_run.side_effect = FileNotFoundError("pacemaker not found")

    result = wrapper.train()

    assert result.success is False

@patch("subprocess.run")
def test_train_failure_subprocess_error(mock_run, training_config, tmp_path):
    """Test training failure due to generic subprocess error."""
    wrapper = PacemakerWrapper(config=training_config, work_dir=tmp_path)

    mock_run.side_effect = subprocess.SubprocessError("Failed to spawn")

    result = wrapper.train()

    assert result.success is False

@patch("subprocess.run")
@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper.check_output")
def test_train_success_no_output(mock_check, mock_run, training_config, tmp_path):
    """Test training succeeds but output file is missing/empty."""
    wrapper = PacemakerWrapper(config=training_config, work_dir=tmp_path)

    mock_run.return_value.returncode = 0
    mock_check.return_value = False # Output check fails

    result = wrapper.train()

    assert result.success is False

def test_train_config_write_permission_error(training_config, tmp_path):
    """Test handling of permission error when writing config."""
    # Make dir read-only
    try:
        tmp_path.chmod(0o500)
        wrapper = PacemakerWrapper(config=training_config, work_dir=tmp_path)

        # This should fail gracefully and return success=False
        result = wrapper.train()
        assert result.success is False
    finally:
        tmp_path.chmod(0o700)
