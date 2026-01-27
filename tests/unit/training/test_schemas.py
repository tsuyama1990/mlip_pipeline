import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.training import TrainingConfig


def test_training_config_valid() -> None:
    """Test valid TrainingConfig creation."""
    config = TrainingConfig(
        cutoff=5.0,
        b_basis_size=100,
        kappa=0.5,
        kappa_f=0.5,
        max_num_epochs=500,
        batch_size=32,
        ladder_step=[10, 20],
    )
    assert config.cutoff == 5.0
    assert config.b_basis_size == 100
    assert config.batch_size == 32
    assert config.ladder_step == [10, 20]


def test_training_config_defaults() -> None:
    """Test default values."""
    config = TrainingConfig(cutoff=5.0, b_basis_size=100, kappa=0.5, kappa_f=0.5, batch_size=32)
    assert config.training_data_path == "data/train.xyz"
    assert config.test_data_path == "data/test.xyz"
    assert config.max_num_epochs == 1000
    assert config.ladder_step == []


def test_training_config_invalid_cutoff() -> None:
    """Test invalid cutoff values."""
    with pytest.raises(ValidationError):
        TrainingConfig(
            cutoff=0.5,  # Too small
            b_basis_size=100,
            kappa=0.5,
            kappa_f=0.5,
            batch_size=32,
        )

    with pytest.raises(ValidationError):
        TrainingConfig(
            cutoff=25.0,  # Too large
            b_basis_size=100,
            kappa=0.5,
            kappa_f=0.5,
            batch_size=32,
        )


def test_training_config_invalid_basis() -> None:
    """Test invalid basis size."""
    with pytest.raises(ValidationError):
        TrainingConfig(cutoff=5.0, b_basis_size=0, kappa=0.5, kappa_f=0.5, batch_size=32)


def test_training_config_extra_forbid() -> None:
    """Test that extra fields are forbidden."""
    with pytest.raises(ValidationError):
        TrainingConfig(
            cutoff=5.0,
            b_basis_size=100,
            kappa=0.5,
            kappa_f=0.5,
            batch_size=32,
            extra_field="invalid",
        )
