from pathlib import Path

import pytest

from mlip_autopipec.training.metrics import LogParser


def test_parse_success(tmp_path: Path) -> None:
    """Test successful log parsing."""
    log_content = """
    Start training...
    Epoch 1
    RMSE (energy): 0.5
    RMSE (forces): 0.05
    Epoch 100
    RMSE (energy): 0.1
    RMSE (forces): 0.01
    """
    log_file = tmp_path / "log.txt"
    log_file.write_text(log_content)

    parser = LogParser()
    metrics = parser.parse_file(log_file)

    assert metrics is not None
    assert metrics.epoch == 100
    assert metrics.rmse_energy == 0.1
    assert metrics.rmse_force == 0.01


def test_parse_divergence_nan(tmp_path: Path) -> None:
    """Test detection of NaN."""
    log_content = """
    Epoch 50
    RMSE (energy): NaN
    RMSE (forces): 0.05
    """
    log_file = tmp_path / "log.txt"
    log_file.write_text(log_content)

    parser = LogParser()
    with pytest.raises(ValueError, match="Training diverged"):
        parser.parse_file(log_file)


def test_parse_empty(tmp_path: Path) -> None:
    """Test empty log."""
    log_file = tmp_path / "log.txt"
    log_file.touch()

    parser = LogParser()
    metrics = parser.parse_file(log_file)
    assert metrics is None


def test_parse_malformed(tmp_path: Path) -> None:
    """Test malformed log."""
    log_content = "Just some text without metrics"
    log_file = tmp_path / "log.txt"
    log_file.write_text(log_content)

    parser = LogParser()
    metrics = parser.parse_file(log_file)
    assert metrics is None


def test_parse_file_not_found(tmp_path: Path) -> None:
    """Test file not found."""
    parser = LogParser()
    metrics = parser.parse_file(tmp_path / "non_existent.log")
    assert metrics is None


def test_parse_float_error(tmp_path: Path) -> None:
    """Test value conversion error."""
    log_content = """
    Epoch 1
    RMSE (energy): invalid
    RMSE (forces): 0.01
    """
    log_file = tmp_path / "log.txt"
    log_file.write_text(log_content)

    parser = LogParser()
    metrics = parser.parse_file(log_file)
    assert metrics is None


def test_parse_infinite(tmp_path: Path) -> None:
    """Test infinite values."""
    log_content = """
    Epoch 1
    RMSE (energy): inf
    RMSE (forces): 0.01
    """
    log_file = tmp_path / "log.txt"
    log_file.write_text(log_content)

    parser = LogParser()
    with pytest.raises(ValueError, match="Training diverged"):
        parser.parse_file(log_file)
