import pytest
from pydantic import ValidationError
from mlip_autopipec.domain_models.production import ProductionManifest
from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric

def test_production_manifest_valid():
    metrics = [ValidationMetric(name="RMSE Energy", value=0.001, passed=True)]
    result = ValidationResult(
        potential_id="test_pot",
        metrics=metrics,
        overall_status="PASS"
    )
    manifest = ProductionManifest(
        version="1.0.0",
        author="Test User",
        training_set_size=100,
        validation_metrics=result
    )
    assert manifest.version == "1.0.0"
    assert manifest.author == "Test User"
    assert manifest.license == "MIT"

def test_production_manifest_invalid_version():
    metrics = [ValidationMetric(name="RMSE Energy", value=0.001, passed=True)]
    result = ValidationResult(
        potential_id="test_pot",
        metrics=metrics,
        overall_status="PASS"
    )
    with pytest.raises(ValidationError):
        ProductionManifest(
            version="1.0", # Invalid SemVer
            author="Test User",
            training_set_size=100,
            validation_metrics=result
        )

def test_production_manifest_invalid_size():
    metrics = [ValidationMetric(name="RMSE Energy", value=0.001, passed=True)]
    result = ValidationResult(
        potential_id="test_pot",
        metrics=metrics,
        overall_status="PASS"
    )
    with pytest.raises(ValidationError):
        ProductionManifest(
            version="1.0.0",
            author="Test User",
            training_set_size=-10,
            validation_metrics=result
        )
