"""Unit tests for MACE configuration."""

import pytest
from pydantic import ValidationError

from pyacemaker.core.config import CONSTANTS, DFTConfig, MaceConfig, OracleConfig

# Bypass file checks for tests
CONSTANTS.skip_file_checks = True


def test_mace_config_defaults() -> None:
    """Test default values for MaceConfig."""
    config = MaceConfig()
    assert config.model_path == "medium"
    assert config.device == "cpu"
    assert config.default_dtype == "float64"
    assert config.batch_size == 32


def test_mace_config_valid() -> None:
    """Test valid MaceConfig initialization."""
    config = MaceConfig(
        model_path="custom.model",
        device="cuda",
        default_dtype="float32",
        batch_size=64,
    )
    assert config.model_path == "custom.model"
    assert config.device == "cuda"
    assert config.default_dtype == "float32"
    assert config.batch_size == 64


def test_mace_config_invalid_device() -> None:
    """Test invalid device raises ValidationError."""
    with pytest.raises(ValidationError) as excinfo:
        MaceConfig(device="invalid_device")
    assert "Invalid device" in str(excinfo.value)


def test_mace_config_invalid_dtype() -> None:
    """Test invalid dtype raises ValidationError."""
    with pytest.raises(ValidationError) as excinfo:
        MaceConfig(default_dtype="int32")
    assert "Invalid dtype" in str(excinfo.value)


def test_oracle_config_with_mace() -> None:
    """Test OracleConfig with MaceConfig."""
    dft_config = DFTConfig(
        code="quantum_espresso",
        command="pw.x",
        pseudopotentials={"Fe": "Fe.pbe.UPF"},
    )
    mace_config = MaceConfig(model_path="test.model")

    # Needs skip_file_checks=True or mock paths for DFT validation
    # DFTConfig validation checks for file existence unless skipped
    # We can mock the path validation or set PYACEMAKER_SKIP_FILE_CHECKS env var
    # Or just rely on the fact that we are not checking files in unit tests if config allows?
    # DFTConfig checks CONSTANTS.skip_file_checks.

    # We will use valid dummy data that passes basic validation
    # But wait, DFTConfig validates pseudopotentials existence.
    # I should set skip_file_checks=True in CONSTANTS or mock Path.exists

    oracle_config = OracleConfig(dft=dft_config, mace=mace_config)
    assert oracle_config.mace is not None
    assert oracle_config.mace.model_path == "test.model"


def test_oracle_config_without_mace() -> None:
    """Test OracleConfig without MaceConfig (optional)."""
    dft_config = DFTConfig(
        code="quantum_espresso",
        command="pw.x",
        pseudopotentials={"Fe": "Fe.pbe.UPF"},
    )
    oracle_config = OracleConfig(dft=dft_config)
    assert oracle_config.mace is None
