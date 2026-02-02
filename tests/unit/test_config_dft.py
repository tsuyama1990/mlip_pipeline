import pytest
from pydantic import ValidationError

from mlip_autopipec.config import DFTConfig


def test_dft_config_defaults() -> None:
    config = DFTConfig(pseudopotentials={"Si": "Si.upf"})
    assert config.command == "pw.x"
    assert config.kspacing == 0.04
    assert config.ecutwfc == 50.0
    assert config.max_retries == 3


def test_dft_config_missing_pseudos() -> None:
    with pytest.raises(ValidationError):
        DFTConfig()  # type: ignore[call-arg]


def test_dft_config_invalid_pseudos() -> None:
    with pytest.raises(ValidationError):
        DFTConfig(pseudopotentials="not a dict")  # type: ignore[arg-type]


def test_dft_config_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        DFTConfig(pseudopotentials={"Si": "Si.upf"}, extra_field="bad")  # type: ignore[call-arg]
