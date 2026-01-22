import pytest
from pathlib import Path
from mlip_autopipec.training.metrics import LogParser
from mlip_autopipec.config.schemas.training import TrainingMetrics

SAMPLE_LOG = """
Pacemaker v1.0
...
Epoch 1
RMSE (energy) : 10.5
RMSE (forces) : 0.5
...
Epoch 10
RMSE (energy) : 2.5
RMSE (forces) : 0.05
Training finished.
"""

def test_parse_log(tmp_path):
    """Test parsing a valid log file."""
    log_file = tmp_path / "log.txt"
    log_file.write_text(SAMPLE_LOG)

    parser = LogParser()
    metrics = parser.parse_file(log_file)

    assert isinstance(metrics, TrainingMetrics)
    assert metrics.epoch == 10
    assert metrics.rmse_energy == 2.5
    assert metrics.rmse_force == 0.05

def test_parse_log_empty(tmp_path):
    """Test parsing an empty or invalid log file."""
    log_file = tmp_path / "empty.txt"
    log_file.write_text("")

    parser = LogParser()
    metrics = parser.parse_file(log_file)

    assert metrics is None

def test_parse_log_missing_file(tmp_path):
    """Test parsing a missing file."""
    log_file = tmp_path / "missing.txt"
    parser = LogParser()
    metrics = parser.parse_file(log_file)
    assert metrics is None

def test_parse_log_malformed_rmse(tmp_path):
    """Test parsing a log with malformed RMSE."""
    log_content = """
    Epoch 1
    RMSE (energy) : NOT_A_NUMBER
    RMSE (forces) : 0.5
    """
    log_file = tmp_path / "malformed.txt"
    log_file.write_text(log_content)

    parser = LogParser()
    metrics = parser.parse_file(log_file)
    assert metrics is None

def test_parse_log_divergence(tmp_path):
    """Test detection of divergence (NaN)."""
    log_content = """
    Epoch 2
    RMSE (energy) : NaN
    RMSE (forces) : NaN
    """
    log_file = tmp_path / "diverge.txt"
    log_file.write_text(log_content)

    parser = LogParser()

    with pytest.raises(ValueError, match="diverged"):
        parser.parse_file(log_file)
