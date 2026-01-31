from pathlib import Path
import pytest
from pydantic import ValidationError
from mlip_autopipec.domain_models.training import TrainingConfig, TrainingResult
from mlip_autopipec.domain_models.job import JobStatus


def test_training_config_defaults():
    config = TrainingConfig()
    assert config.batch_size == 100
    assert config.max_epochs == 100
    assert config.ladder_step == [100, 10]
    assert config.kappa == 0.3
    assert config.active_set_optimization is True
    assert config.initial_potential is None


def test_training_config_validation():
    # Valid
    TrainingConfig(batch_size=1, max_epochs=1, kappa=0.0)
    TrainingConfig(batch_size=1, max_epochs=1, kappa=1.0)

    # Invalid batch_size
    with pytest.raises(ValidationError):
        TrainingConfig(batch_size=0)

    # Invalid max_epochs
    with pytest.raises(ValidationError):
        TrainingConfig(max_epochs=-1)

    # Invalid kappa
    with pytest.raises(ValidationError):
        TrainingConfig(kappa=-0.1)
    with pytest.raises(ValidationError):
        TrainingConfig(kappa=1.1)


def test_training_result():
    result = TrainingResult(
        job_id="test_job",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp/work"),
        duration_seconds=10.0,
        log_content="log",
        potential_path=Path("/tmp/model.yace"),
        validation_metrics={"rmse_energy": 0.001},
    )
    assert result.potential_path == Path("/tmp/model.yace")
    assert result.validation_metrics["rmse_energy"] == 0.001


def test_training_result_extra_forbidden():
    with pytest.raises(ValidationError):
        TrainingResult(
            job_id="test_job",
            status=JobStatus.COMPLETED,
            work_dir=Path("/tmp/work"),
            duration_seconds=10.0,
            log_content="log",
            potential_path=Path("/tmp/model.yace"),
            validation_metrics={},
            extra="fail",  # type: ignore
        )
