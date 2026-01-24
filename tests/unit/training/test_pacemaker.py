from pathlib import Path
from unittest.mock import MagicMock, patch
import subprocess

import pytest
import yaml
from ase import Atoms

from mlip_autopipec.config.schemas.training import TrainingConfig, TrainingResult
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


def test_generate_config(wrapper: PacemakerWrapper, training_config: TrainingConfig) -> None:
    """Test YAML config generation."""
    config_path = wrapper.generate_config()

    assert config_path.exists()

    with config_path.open("r") as f:
        data = yaml.safe_load(f)

    assert data["cutoff"] == 4.0
    assert data["fit"]["optimizer"]["max_iter"] == 100
    assert data["fit"]["optimizer"]["batch_size"] == 32
    assert data["ladder_step"] == [10, 50]
    assert data["b_basis"]["size"] == 10


def test_generate_config_fail(wrapper: PacemakerWrapper) -> None:
    """Test config generation failure (e.g. permission error)."""
    with patch.object(Path, "open", side_effect=OSError("Permission denied")), pytest.raises(OSError, match="Permission denied"):
         wrapper.generate_config()


@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper._resolve_executable")
@patch("mlip_autopipec.training.pacemaker.subprocess.run")
def test_train_success(mock_run: MagicMock, mock_resolve: MagicMock, wrapper: PacemakerWrapper, tmp_path: Path) -> None:
    """Test successful training execution."""
    mock_resolve.return_value = "/bin/pacemaker"

    output_file = tmp_path / "output.yace"
    output_file.write_text("fake potential content")

    def side_effect(*args, **kwargs):
        if 'stdout' in kwargs and hasattr(kwargs['stdout'], 'write'):
            kwargs['stdout'].write("Epoch 100\nRMSE (energy): 0.1\nRMSE (forces): 0.01")
        return MagicMock(returncode=0)

    mock_run.side_effect = side_effect

    result = wrapper.train()

    assert result.success
    assert result.metrics is not None
    assert result.metrics.rmse_energy == 0.1
    assert result.potential_path == str(output_file)


@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper._resolve_executable")
@patch("mlip_autopipec.training.pacemaker.subprocess.run")
def test_train_with_initial_potential(mock_run: MagicMock, mock_resolve: MagicMock, wrapper: PacemakerWrapper, tmp_path: Path) -> None:
    """Test training with initial potential."""
    mock_resolve.return_value = "/bin/pacemaker"
    (tmp_path / "output.yace").write_text("fake potential")

    def side_effect(*args, **kwargs):
        if 'stdout' in kwargs and hasattr(kwargs['stdout'], 'write'):
            kwargs['stdout'].write("Epoch 100\nRMSE (energy): 0.1\nRMSE (forces): 0.01")
        return MagicMock(returncode=0)
    mock_run.side_effect = side_effect

    initial_pot = "init.yace"

    result = wrapper.train(initial_potential=initial_pot)

    assert result.success
    args = mock_run.call_args[0][0]
    assert initial_pot in args or any(initial_pot in a for a in args)


@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper._resolve_executable")
@patch("mlip_autopipec.training.pacemaker.subprocess.run")
def test_train_failure_code(mock_run: MagicMock, mock_resolve: MagicMock, wrapper: PacemakerWrapper) -> None:
    """Test training failure (non-zero exit)."""
    mock_resolve.return_value = "/bin/pacemaker"
    mock_run.return_value.returncode = 1

    result = wrapper.train()
    assert not result.success


@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper._resolve_executable")
@patch("mlip_autopipec.training.pacemaker.subprocess.run")
def test_train_failure_no_output(mock_run: MagicMock, mock_resolve: MagicMock, wrapper: PacemakerWrapper, tmp_path: Path) -> None:
    """Test training failure (no output file)."""
    mock_resolve.return_value = "/bin/pacemaker"

    # Simulate success but no file created
    mock_run.return_value.returncode = 0
    # Ensure no yace files
    for f in tmp_path.glob("*.yace"):
        f.unlink()

    result = wrapper.train()
    assert not result.success


@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper._resolve_executable")
@patch("mlip_autopipec.training.pacemaker.subprocess.run")
def test_train_exception(mock_run: MagicMock, mock_resolve: MagicMock, wrapper: PacemakerWrapper) -> None:
    """Test training exception."""
    mock_resolve.return_value = "/bin/pacemaker"
    mock_run.side_effect = subprocess.SubprocessError("Boom")

    result = wrapper.train()
    assert not result.success

@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper._resolve_executable")
@patch("mlip_autopipec.training.pacemaker.subprocess.run")
def test_train_timeout(mock_run: MagicMock, mock_resolve: MagicMock, wrapper: PacemakerWrapper) -> None:
    """Test training timeout."""
    mock_resolve.return_value = "/bin/pacemaker"
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="pacemaker", timeout=10)

    result = wrapper.train()
    assert not result.success

@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper._resolve_executable")
@patch("mlip_autopipec.training.pacemaker.subprocess.run")
def test_train_oserror(mock_run: MagicMock, mock_resolve: MagicMock, wrapper: PacemakerWrapper) -> None:
    """Test training os error."""
    mock_resolve.return_value = "/bin/pacemaker"
    mock_run.side_effect = OSError("Disk full")

    result = wrapper.train()
    assert not result.success


@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper._resolve_executable")
@patch("mlip_autopipec.training.pacemaker.subprocess.run")
def test_select_active_set(mock_run: MagicMock, mock_resolve: MagicMock, wrapper: PacemakerWrapper) -> None:
    """Test active set selection."""
    mock_resolve.return_value = "/bin/pace_activeset"

    candidates = [Atoms("H"), Atoms("He")]
    current_potential = "current.yace"

    mock_run.return_value.stdout = "Selected indices: 0 1\n"
    mock_run.return_value.returncode = 0
    mock_run.side_effect = None

    indices = wrapper.select_active_set(candidates, current_potential)

    assert indices == [0, 1]


@patch("mlip_autopipec.training.pacemaker.PacemakerWrapper._resolve_executable")
@patch("mlip_autopipec.training.pacemaker.subprocess.run")
def test_select_active_set_fail(mock_run: MagicMock, mock_resolve: MagicMock, wrapper: PacemakerWrapper) -> None:
    """Test active set selection failure."""
    mock_resolve.return_value = "/bin/pace_activeset"
    mock_run.return_value.returncode = 1
    mock_run.return_value.stderr = "Error"

    candidates = [Atoms("H")]

    with pytest.raises(RuntimeError):
        wrapper.select_active_set(candidates, "pot.yace")


def test_prepare_data_from_stream(wrapper: PacemakerWrapper, tmp_path: Path) -> None:
    """Test data preparation."""
    data = [Atoms("H")]
    path = wrapper.prepare_data_from_stream(data, "stream.xyz")

    assert path.exists()
    assert path.stat().st_size > 0


def test_prepare_data_from_stream_fail(wrapper: PacemakerWrapper, tmp_path: Path) -> None:
    """Test data preparation failure (unlink fail)."""
    data = [Atoms("H")]
    (tmp_path / "fail.xyz").mkdir()

    # IsADirectoryError is subclass of OSError
    with pytest.raises(OSError):
        wrapper.prepare_data_from_stream(data, "fail.xyz")

def test_prepare_data_from_stream_write_fail(wrapper: PacemakerWrapper, tmp_path: Path) -> None:
    """Test data preparation failure (write fail)."""
    data = [Atoms("H")]
    # mock ase.io.write
    with patch("mlip_autopipec.training.pacemaker.write", side_effect=Exception("Write fail")), pytest.raises(Exception, match="Write fail"):
         wrapper.prepare_data_from_stream(data, "stream_fail.xyz")


def test_resolve_executable_not_found(wrapper: PacemakerWrapper) -> None:
    """Test resolve executable not found."""
    with patch("shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError):
            wrapper._resolve_executable("missing_exe")


def test_resolve_executable_invalid(wrapper: PacemakerWrapper, tmp_path: Path) -> None:
    """Test resolve executable invalid."""
    bad_exe = tmp_path / "bad.exe"
    bad_exe.touch()

    with patch("shutil.which", return_value=str(bad_exe)), pytest.raises(ValueError, match="not a valid executable"):
         wrapper._resolve_executable("bad.exe")

def test_check_output(wrapper: PacemakerWrapper, tmp_path: Path) -> None:
    """Test check_output."""
    f = tmp_path / "test.yace"
    assert not wrapper.check_output(f)

    f.touch()
    assert not wrapper.check_output(f) # Empty

    f.write_text("content")
    assert wrapper.check_output(f)
