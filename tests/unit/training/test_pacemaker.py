from pathlib import Path
from unittest.mock import MagicMock, patch
import subprocess

import pytest
import yaml
from ase import Atoms

from mlip_autopipec.config.schemas.training import TrainingConfig
from mlip_autopipec.training.pacemaker import PacemakerWrapper


@pytest.fixture
def training_config() -> TrainingConfig:
    return TrainingConfig(
        cutoff=4.0,
        b_basis_size=10,
        kappa=0.4,
        kappa_f=0.6,
        batch_size=32,
        max_num_epochs=100,
        ladder_step=[10, 50]
    )


@pytest.fixture
def wrapper(training_config: TrainingConfig, tmp_path: Path) -> PacemakerWrapper:
    return PacemakerWrapper(training_config, tmp_path)


def test_generate_config_unsafe_path(wrapper: PacemakerWrapper) -> None:
    """Test config generation with unsafe paths."""
    wrapper.config.training_data_path = "/etc/passwd"

    with patch("mlip_autopipec.training.pacemaker.validate_path_safety") as mock_validate:
        wrapper.generate_config()
        assert mock_validate.call_count >= 1


def test_prepare_data_unsafe_filename(wrapper: PacemakerWrapper, tmp_path: Path) -> None:
    """Test prepare data with unsafe filename."""
    data = [Atoms("H")]
    with pytest.raises(ValueError, match="Invalid filename"):
        wrapper.prepare_data_from_stream(data, "../unsafe.xyz")


@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper._resolve_executable")
@patch("mlip_autopipec.training.pacemaker.subprocess.run")
def test_train_unsafe_initial_potential(mock_run: MagicMock, mock_resolve: MagicMock, wrapper: PacemakerWrapper) -> None:
    """Test training with non-existent initial potential."""
    mock_resolve.return_value = "/bin/pacemaker"

    with pytest.raises(FileNotFoundError):
        wrapper.train(initial_potential="ghost.yace")


@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper._resolve_executable")
@patch("mlip_autopipec.training.pacemaker.subprocess.run")
def test_select_active_set_unsafe_path(mock_run: MagicMock, mock_resolve: MagicMock, wrapper: PacemakerWrapper) -> None:
    """Test active set selection with unsafe/missing potential."""
    mock_resolve.return_value = "/bin/pace_activeset"

    candidates = [Atoms("H")]

    with pytest.raises(FileNotFoundError):
         wrapper.select_active_set(candidates, "missing.yace")


def test_generate_config(wrapper: PacemakerWrapper, training_config: TrainingConfig) -> None:
    """Test YAML config generation."""
    config_path = wrapper.generate_config()
    assert config_path.exists()
    with config_path.open("r") as f:
        data = yaml.safe_load(f)
    assert data["cutoff"] == 4.0

def test_generate_config_fail(wrapper: PacemakerWrapper) -> None:
    """Test config generation failure (e.g. permission error)."""
    with patch.object(Path, "open", side_effect=OSError("Permission denied")), pytest.raises(OSError, match="Permission denied"):
         wrapper.generate_config()

@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper._resolve_executable")
@patch("mlip_autopipec.training.pacemaker.subprocess.run")
def test_train_success(mock_run: MagicMock, mock_resolve: MagicMock, wrapper: PacemakerWrapper, tmp_path: Path) -> None:
    mock_resolve.return_value = "/bin/pacemaker"
    output_file = tmp_path / "output.yace"
    output_file.write_text("content")

    def side_effect(*args, **kwargs):
        if 'stdout' in kwargs and hasattr(kwargs['stdout'], 'write'):
            kwargs['stdout'].write("Epoch 100\nRMSE (energy): 0.1\nRMSE (forces): 0.01")
        return MagicMock(returncode=0)
    mock_run.side_effect = side_effect

    result = wrapper.train()
    assert result.success

@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper._resolve_executable")
@patch("mlip_autopipec.training.pacemaker.subprocess.run")
def test_select_active_set_success(mock_run: MagicMock, mock_resolve: MagicMock, wrapper: PacemakerWrapper, tmp_path: Path) -> None:
    mock_resolve.return_value = "/bin/pace_activeset"
    mock_run.return_value.stdout = "Selected indices: 0 1"
    mock_run.return_value.returncode = 0
    mock_run.side_effect = None

    (tmp_path / "current.yace").touch()

    candidates = [Atoms("H")]
    indices = wrapper.select_active_set(candidates, tmp_path / "current.yace")
    assert indices == [0, 1]

@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper._resolve_executable")
@patch("mlip_autopipec.training.pacemaker.subprocess.run")
def test_train_with_initial_potential(mock_run: MagicMock, mock_resolve: MagicMock, wrapper: PacemakerWrapper, tmp_path: Path) -> None:
    mock_resolve.return_value = "/bin/pacemaker"
    (tmp_path / "output.yace").write_text("fake potential")
    (tmp_path / "init.yace").touch()

    def side_effect(*args, **kwargs):
        if 'stdout' in kwargs and hasattr(kwargs['stdout'], 'write'):
            kwargs['stdout'].write("Epoch 100\nRMSE (energy): 0.1\nRMSE (forces): 0.01")
        return MagicMock(returncode=0)
    mock_run.side_effect = side_effect

    result = wrapper.train(initial_potential=tmp_path / "init.yace")
    assert result.success

@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper._resolve_executable")
@patch("mlip_autopipec.training.pacemaker.subprocess.run")
def test_train_failure_code(mock_run: MagicMock, mock_resolve: MagicMock, wrapper: PacemakerWrapper) -> None:
    mock_resolve.return_value = "/bin/pacemaker"
    mock_run.return_value.returncode = 1
    result = wrapper.train()
    assert not result.success

@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper._resolve_executable")
@patch("mlip_autopipec.training.pacemaker.subprocess.run")
def test_train_exception(mock_run: MagicMock, mock_resolve: MagicMock, wrapper: PacemakerWrapper) -> None:
    mock_resolve.return_value = "/bin/pacemaker"
    mock_run.side_effect = subprocess.SubprocessError("Boom")
    result = wrapper.train()
    assert not result.success

def test_resolve_executable_not_found(wrapper: PacemakerWrapper) -> None:
    with patch("shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError):
            wrapper._resolve_executable("missing_exe")

def test_check_output(wrapper: PacemakerWrapper, tmp_path: Path) -> None:
    f = tmp_path / "test.yace"
    assert not wrapper.check_output(f)
    f.touch()
    assert not wrapper.check_output(f)
    f.write_text("content")
    assert wrapper.check_output(f)
