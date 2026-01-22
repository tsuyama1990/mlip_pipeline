import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.dft import DFTConfig


def test_dft_config_valid(tmp_path):
    # Create valid directory
    (tmp_path / "pseudos").mkdir()

    config = DFTConfig(
        pseudopotential_dir=tmp_path / "pseudos",
        ecutwfc=60.0,
        kspacing=0.04,
        nspin=1,
        mixing_beta=0.7,
        diagonalization="david",
        smearing="mv",
        degauss=0.02
    )
    assert config.ecutwfc == 60.0
    assert config.mixing_beta == 0.7
    assert config.command == "pw.x"  # Default

def test_dft_config_defaults(tmp_path):
    (tmp_path / "pseudos").mkdir()
    config = DFTConfig(
        pseudopotential_dir=tmp_path / "pseudos",
        ecutwfc=60.0,
        kspacing=0.04
    )
    assert config.mixing_beta == 0.7
    assert config.diagonalization == "david"
    assert config.smearing == "mv"
    assert config.degauss == 0.02
    assert config.recoverable is True
    assert config.max_retries == 5
    assert config.timeout == 3600.0

def test_dft_config_validation_errors(tmp_path):
    # Invalid directory
    with pytest.raises(ValidationError, match="does not exist"):
        DFTConfig(
            pseudopotential_dir=tmp_path / "nonexistent",
            ecutwfc=60.0,
            kspacing=0.04
        )

    # Valid dir for subsequent tests
    (tmp_path / "pseudos").mkdir()
    valid_dir = tmp_path / "pseudos"

    with pytest.raises(ValidationError):
        DFTConfig(
            pseudopotential_dir=valid_dir,
            ecutwfc=-10.0, # Invalid
            kspacing=0.04
        )

    with pytest.raises(ValidationError):
        DFTConfig(
            pseudopotential_dir=valid_dir,
            ecutwfc=60.0,
            kspacing=0.04,
            mixing_beta=1.5 # Invalid
        )

    with pytest.raises(ValidationError):
        DFTConfig(
            pseudopotential_dir=valid_dir,
            ecutwfc=60.0,
            kspacing=0.04,
            diagonalization="invalid" # Invalid
        )

    with pytest.raises(ValidationError):
        DFTConfig(
            pseudopotential_dir=valid_dir,
            ecutwfc=60.0,
            kspacing=0.04,
            nspin=3 # Invalid
        )
