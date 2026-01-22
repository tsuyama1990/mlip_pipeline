import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.training import TrainingConfig, TrainingMetrics

def test_training_config_valid():
    """Test creating a valid TrainingConfig."""
    config = TrainingConfig(
        cutoff=5.0,
        b_basis_size=1000,
        kappa=0.5,
        kappa_f=100.0,
        max_iter=500
    )
    assert config.cutoff == 5.0
    assert config.b_basis_size == 1000
    assert config.kappa == 0.5
    assert config.kappa_f == 100.0
    assert config.max_iter == 500
    assert config.training_data_path == "data/train.xyz"

def test_training_config_invalid_cutoff():
    """Test validation fails for non-positive cutoff."""
    with pytest.raises(ValidationError):
        TrainingConfig(
            cutoff=-1.0,
            b_basis_size=1000,
            kappa=0.5,
            kappa_f=100.0
        )

def test_training_config_invalid_basis():
    """Test validation fails for non-positive basis size."""
    with pytest.raises(ValidationError):
        TrainingConfig(
            cutoff=5.0,
            b_basis_size=0,
            kappa=0.5,
            kappa_f=100.0
        )

def test_training_config_extra_forbid():
    """Test that extra fields are forbidden."""
    with pytest.raises(ValidationError):
        TrainingConfig(
            cutoff=5.0,
            b_basis_size=1000,
            kappa=0.5,
            kappa_f=100.0,
            extra_field="illegal"
        )

def test_training_metrics_valid():
    """Test creating a valid TrainingMetrics."""
    metrics = TrainingMetrics(
        epoch=10,
        rmse_energy=2.5,
        rmse_force=0.01
    )
    assert metrics.epoch == 10
    assert metrics.rmse_energy == 2.5
    assert metrics.rmse_force == 0.01

def test_training_metrics_invalid():
    """Test validation for TrainingMetrics."""
    with pytest.raises(ValidationError):
        TrainingMetrics(epoch=-1, rmse_energy=1.0, rmse_force=1.0)

    with pytest.raises(ValidationError):
        TrainingMetrics(epoch=1, rmse_energy=-1.0, rmse_force=1.0)
