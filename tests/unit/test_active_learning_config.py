import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import ActiveLearningConfig, GlobalConfig


def test_active_learning_config_defaults():
    config = ActiveLearningConfig()
    assert config.perturbation_magnitude == 0.1
    assert config.n_candidates == 20
    assert config.sampling_method == "perturbation"
    assert config.max_retries == 3

def test_active_learning_config_validation():
    # Valid
    config = ActiveLearningConfig(perturbation_magnitude=0.5, n_candidates=10)
    assert config.perturbation_magnitude == 0.5

    # Invalid magnitude
    with pytest.raises(ValidationError):
        ActiveLearningConfig(perturbation_magnitude=0.0)

    # Invalid candidates
    with pytest.raises(ValidationError):
        ActiveLearningConfig(n_candidates=0)

    # Invalid retries
    with pytest.raises(ValidationError):
        ActiveLearningConfig(max_retries=-1)

def test_global_config_integration(tmp_path):
    # Ensure it's part of GlobalConfig and has defaults
    # We need to provide required fields for GlobalConfig (OrchestratorConfig.work_dir)
    from mlip_autopipec.domain_models.config import OrchestratorConfig

    OrchestratorConfig(work_dir=tmp_path)

    assert "active_learning" in GlobalConfig.model_fields
    assert GlobalConfig.model_fields["active_learning"].default_factory is not None
