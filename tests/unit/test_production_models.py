from datetime import datetime

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.production import ProductionManifest


def test_production_manifest_valid() -> None:
    manifest = ProductionManifest(
        version="1.0.0",
        author="Test User",
        training_set_size=100,
        validation_metrics={"rmse_energy": 0.001, "rmse_forces": 0.01},
    )
    assert manifest.version == "1.0.0"
    assert manifest.author == "Test User"
    assert manifest.training_set_size == 100
    assert manifest.validation_metrics["rmse_energy"] == 0.001
    assert isinstance(manifest.creation_date, datetime)


def test_production_manifest_invalid_version() -> None:
    with pytest.raises(ValidationError):
        ProductionManifest(
            version=1.0,  # type: ignore[arg-type]
            author="Test User",
            training_set_size=100,
        )


def test_production_manifest_invalid_training_set_size() -> None:
    with pytest.raises(ValidationError):
        ProductionManifest(
            version="1.0.0",
            author="Test User",
            training_set_size=-10,  # Should be ge=0
        )


def test_production_manifest_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ProductionManifest(
            version="1.0.0",
            author="Test User",
            training_set_size=100,
            extra_field="Not allowed",  # type: ignore[call-arg]
        )
