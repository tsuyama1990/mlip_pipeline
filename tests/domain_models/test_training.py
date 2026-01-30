from pathlib import Path
from mlip_autopipec.domain_models.training import TrainingConfig, TrainingResult
from mlip_autopipec.domain_models.job import JobStatus


def test_training_config_defaults():
    """Test default values for TrainingConfig."""
    config = TrainingConfig()
    assert config.batch_size == 100
    assert config.max_epochs == 100
    assert config.ladder_step == [100, 10]
    assert config.kappa == 0.5
    assert config.initial_potential is None
    assert config.active_set_optimization is True


def test_training_config_custom():
    """Test custom values for TrainingConfig."""
    config = TrainingConfig(
        batch_size=50,
        max_epochs=200,
        ladder_step=[50, 5],
        kappa=0.1,
        initial_potential=Path("old.yace"),
        active_set_optimization=False
    )
    assert config.batch_size == 50
    assert config.initial_potential == Path("old.yace")
    assert config.active_set_optimization is False


def test_training_result_valid():
    """Test creating a valid TrainingResult."""
    res = TrainingResult(
        job_id="test_job",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp"),
        duration_seconds=50.0,
        log_content="Log...",
        potential_path=Path("new.yace"),
        validation_metrics={"rmse_energy": 0.002, "rmse_force": 0.01},
    )
    assert res.potential_path == Path("new.yace")
    assert res.validation_metrics["rmse_energy"] == 0.002
    assert res.duration_seconds == 50.0
