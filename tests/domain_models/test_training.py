from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models import JobStatus, TrainingConfig, TrainingResult


def test_training_config_defaults():
    config = TrainingConfig()
    assert config.batch_size == 100
    assert config.max_epochs == 100
    assert config.ladder_step == [100, 10]
    assert config.kappa == 0.4
    assert config.initial_potential is None
    assert config.active_set_optimization is True


def test_training_config_valid():
    config = TrainingConfig(
        batch_size=50,
        max_epochs=10,
        ladder_step=[20],
        kappa=0.5,
        initial_potential=Path("old.yace"),
        active_set_optimization=False,
    )
    assert config.batch_size == 50
    assert config.max_epochs == 10
    assert config.ladder_step == [20]
    assert config.kappa == 0.5
    assert config.initial_potential == Path("old.yace")
    assert config.active_set_optimization is False


def test_training_config_extra_forbid():
    with pytest.raises(ValidationError):
        TrainingConfig(extra_field="fail")  # type: ignore


def test_training_result_valid():
    res = TrainingResult(
        job_id="train_01",
        status=JobStatus.COMPLETED,
        work_dir=Path("work"),
        duration_seconds=10.0,
        log_content="log",
        potential_path=Path("pot.yace"),
        validation_metrics={"energy_rmse": 0.005},
    )
    assert res.potential_path == Path("pot.yace")
    assert res.validation_metrics["energy_rmse"] == 0.005


def test_training_result_extra_forbid():
    with pytest.raises(ValidationError):
        TrainingResult(
            job_id="train_01",
            status=JobStatus.COMPLETED,
            work_dir=Path("work"),
            duration_seconds=10.0,
            log_content="log",
            potential_path=Path("pot.yace"),
            validation_metrics={"energy_rmse": 0.005},
            extra_field="fail",  # type: ignore
        )
