import pytest
from pydantic import ValidationError
from mlip_autopipec.domain_models.training import TrainingConfig, TrainingResult
from mlip_autopipec.domain_models.config import PotentialConfig
from pathlib import Path

def test_training_config_valid():
    config = TrainingConfig(
        batch_size=50,
        max_epochs=200,
        ladder_step=[50, 20],
        kappa=0.5,
        active_set_optimization=False
    )
    assert config.batch_size == 50
    assert config.active_set_optimization is False

def test_training_result_schema():
    res = TrainingResult(
        job_id="train_1",
        status="COMPLETED",
        work_dir=Path("work"),
        duration_seconds=10.5,
        log_content="log",
        potential_path=Path("pot.yace"),
        validation_metrics={"rmse_energy": 0.001}
    )
    assert res.validation_metrics["rmse_energy"] == 0.001
