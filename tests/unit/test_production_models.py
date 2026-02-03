from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.production import ProductionManifest


def test_production_manifest_valid() -> None:
    """Test valid manifest creation."""
    data: dict[str, Any] = {
        "version": "1.0.0",
        "author": "Jane Doe",
        "training_set_size": 500,
        "validation_metrics": {"rmse_e": 0.001, "rmse_f": 0.05},
        "creation_date": datetime.now(UTC),
    }
    manifest = ProductionManifest(**data)
    assert manifest.version == "1.0.0"
    assert manifest.author == "Jane Doe"
    assert manifest.training_set_size == 500
    assert manifest.validation_metrics["rmse_e"] == 0.001


def test_production_manifest_defaults() -> None:
    """Test default creation_date."""
    data: dict[str, Any] = {
        "version": "1.0.0",
        "author": "Jane Doe",
        "training_set_size": 500,
        "validation_metrics": {"rmse_e": 0.001},
    }
    manifest = ProductionManifest(**data)
    assert isinstance(manifest.creation_date, datetime)


def test_production_manifest_extra_forbid() -> None:
    """Test strict config."""
    data: dict[str, Any] = {
        "version": "1.0.0",
        "author": "Jane Doe",
        "training_set_size": 500,
        "validation_metrics": {},
        "extra_field": "forbidden",
    }
    with pytest.raises(ValidationError):
        ProductionManifest(**data)


def test_production_manifest_types() -> None:
    """Test strict typing."""
    data: dict[str, Any] = {
        "version": "1.0.0",
        "author": "Jane Doe",
        "training_set_size": "five hundred",  # Invalid type
        "validation_metrics": {},
    }
    with pytest.raises(ValidationError):
        ProductionManifest(**data)
