import pytest
from pydantic import ValidationError

from mlip_autopipec.config.models import (
    Composition,
    ExplorerConfig,
    FingerprintConfig,
    Resources,
    TargetSystem,
)


def test_resources_validation():
    """Test Resources model validation."""
    # Valid
    res = Resources(dft_code="quantum_espresso", parallel_cores=4)
    assert res.parallel_cores == 4

    # Invalid cores
    with pytest.raises(ValidationError):
        Resources(dft_code="quantum_espresso", parallel_cores=0)

    # Invalid code
    with pytest.raises(ValidationError):
        Resources(dft_code="bad_code", parallel_cores=4)


def test_target_system_validation():
    """Test TargetSystem validation."""
    # Valid
    TargetSystem(elements=["Fe", "Ni"], composition=Composition({"Fe": 0.5, "Ni": 0.5}))

    # Invalid element in list
    with pytest.raises(ValidationError) as exc:
        TargetSystem(elements=["Xy"], composition=Composition({"Xy": 1.0}))
    assert "not a valid chemical symbol" in str(exc.value)

    # Mismatch keys
    with pytest.raises(ValidationError) as exc:
        TargetSystem(elements=["Fe"], composition=Composition({"Fe": 0.5, "Ni": 0.5}))
    assert "Composition keys must match" in str(exc.value)


def test_fingerprint_config_validation():
    """Test FingerprintConfig validation."""
    # Valid
    fp = FingerprintConfig(species=["Al", "Cu"])
    assert fp.soap_rcut == 5.0  # default

    # Invalid species
    with pytest.raises(ValidationError):
        FingerprintConfig(species=[])


def test_explorer_config_validation():
    """Test ExplorerConfig validation."""
    fp = FingerprintConfig(species=["Al"])

    # Valid
    ExplorerConfig(surrogate_model_path="model.pt", max_force_threshold=10.0, fingerprint=fp)

    # Invalid threshold
    with pytest.raises(ValidationError):
        ExplorerConfig(surrogate_model_path="model.pt", max_force_threshold=-5.0, fingerprint=fp)
