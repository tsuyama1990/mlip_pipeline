"""Unit tests for the system-level Pydantic configuration schemas."""

import pytest
from pydantic import ValidationError

from mlip_autopipec.config.system import (
    DFTControlParams,
    DFTElectronsParams,
    DFTParams,
    DFTSystemParams,
    Pseudopotentials,
    SystemConfig,
)


def test_dft_control_params_defaults() -> None:
    """Test that DFTControlParams has correct default values."""
    params = DFTControlParams()
    assert params.calculation == "scf"
    assert params.verbosity == "high"


def test_dft_system_params_validation() -> None:
    """Test validation rules for DFTSystemParams."""
    with pytest.raises(ValidationError):
        DFTSystemParams(nat=0, ntyp=1)  # nat must be >= 1

    with pytest.raises(ValidationError):
        DFTSystemParams(nat=1, ntyp=0)  # ntyp must be >= 1


def test_dft_electrons_params_defaults() -> None:
    """Test that DFTElectronsParams has correct default values."""
    params = DFTElectronsParams()
    assert params.mixing_beta == 0.7
    assert params.conv_thr == 1.0e-10


def test_pseudopotentials_valid() -> None:
    """Test that a valid Pseudopotentials model is parsed correctly."""
    data = {
        "Ni": "Ni.pbe-n-rrkjus_psl.1.0.0.UPF",
        "Fe": "Fe.pbe-n-rrkjus_psl.1.0.0.UPF",
    }
    pseudos = Pseudopotentials(root=data)
    assert pseudos.root["Ni"] == "Ni.pbe-n-rrkjus_psl.1.0.0.UPF"


def test_pseudopotentials_invalid_element() -> None:
    """Test that an invalid chemical symbol in Pseudopotentials raises an error."""
    data = {"Xx": "Xx.pbe-n-rrkjus_psl.1.0.0.UPF"}
    with pytest.raises(ValidationError, match="'Xx' is not a valid chemical symbol"):
        Pseudopotentials(root=data)


def test_system_config_valid() -> None:
    """Test a valid SystemConfig instantiation."""
    dft_params = {
        "pseudopotentials": {"Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF"},
        "system": {"nat": 1, "ntyp": 1},
    }
    config = SystemConfig(dft=dft_params, db_path="test.db")
    assert config.db_path == "test.db"
    assert isinstance(config.dft, DFTParams)
    assert config.dft.control.calculation == "scf"


def test_system_config_extra_field_forbidden() -> None:
    """Test that an extra, undefined field in SystemConfig raises an error."""
    dft_params = {
        "pseudopotentials": {"Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF"},
        "system": {"nat": 1, "ntyp": 1},
    }
    with pytest.raises(ValidationError) as exc_info:
        SystemConfig(
            dft=dft_params,
            db_path="test.db",
            extra_param="should_fail",  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)
