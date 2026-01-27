import pytest
from pydantic import ValidationError
from pathlib import Path
from mlip_autopipec.config.schemas.dft import DFTConfig

def test_dft_config_valid(tmp_path):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()

    config_data = {
        "command": "mpirun -np 4 pw.x",
        "pseudopotential_dir": pseudo_dir,
        "pseudopotentials": {"Si": "Si.upf"},
        "kspacing": 0.04
    }
    config = DFTConfig(**config_data)
    assert config.command == "mpirun -np 4 pw.x"
    assert config.kspacing == 0.04
    assert config.pseudopotentials["Si"] == "Si.upf"

def test_dft_config_invalid_command(tmp_path):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    config_data = {
        "command": "pw.x; rm -rf /",
        "pseudopotential_dir": pseudo_dir,
        "pseudopotentials": {"Si": "Si.upf"}
    }

    with pytest.raises(ValidationError) as excinfo:
        DFTConfig(**config_data)
    assert "insecure characters" in str(excinfo.value)

def test_dft_config_invalid_path(tmp_path):
    non_existent = tmp_path / "nowhere"
    config_data = {
        "command": "pw.x",
        "pseudopotential_dir": non_existent,
        "pseudopotentials": {"Si": "Si.upf"}
    }
    with pytest.raises(ValidationError) as excinfo:
        DFTConfig(**config_data)
    assert "does not exist" in str(excinfo.value)

def test_dft_config_invalid_element(tmp_path):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()

    config_data = {
        "command": "pw.x",
        "pseudopotential_dir": pseudo_dir,
        "pseudopotentials": {"Xz": "Xz.upf"} # Invalid element
    }
    with pytest.raises(ValidationError) as excinfo:
        DFTConfig(**config_data)
    assert "Invalid chemical symbol" in str(excinfo.value)
