"""
Tests for configuration schemas.
"""
from pathlib import Path
import pytest
from pydantic import ValidationError
import yaml

from mlip_autopipec.core.config import DFTConfig, GlobalConfig, AppConfig, load_config
from mlip_autopipec.core.exceptions import ConfigError

def test_dft_config_defaults():
    """Verify default values for DFTConfig."""
    config = DFTConfig(
        command="pw.x",
        pseudopotential_dir=Path("/tmp")
    )
    assert config.kpoints_density == 0.15
    assert config.scf_convergence_threshold == 1e-6
    assert config.mixing_beta == 0.7
    assert config.smearing == "mv"

def test_dft_config_validation():
    """Verify validation constraints."""
    # Invalid convergence threshold
    with pytest.raises(ValidationError):
        DFTConfig(
            command="pw.x",
            pseudopotential_dir=Path("/tmp"),
            scf_convergence_threshold=0.0
        )

    # Invalid mixing beta
    with pytest.raises(ValidationError):
        DFTConfig(
            command="pw.x",
            pseudopotential_dir=Path("/tmp"),
            mixing_beta=1.5
        )

def test_load_config_valid(tmp_path):
    """Verify loading a valid configuration file."""
    config_data = {
        "global": {
            "project_name": "TestProject",
            "database_path": str(tmp_path / "test.db"),
            "logging_level": "DEBUG"
        },
        "dft": {
            "code": "quantum_espresso",
            "command": "mpirun -np 2 pw.x",
            "pseudopotential_dir": str(tmp_path / "pseudos"),
            "scf_convergence_threshold": 1e-8
        }
    }

    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Create pseudo dir
    (tmp_path / "pseudos").mkdir()

    config = load_config(config_file)

    assert config.global_config.project_name == "TestProject"
    assert config.dft_config.scf_convergence_threshold == 1e-8
    assert config.dft_config.is_valid_pseudopotential_dir

def test_load_config_missing_file():
    """Verify error when file is missing."""
    with pytest.raises(ConfigError, match="not found"):
        load_config(Path("non_existent.yaml"))

def test_load_config_invalid_yaml(tmp_path):
    """Verify error when YAML is invalid."""
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("global: { unclosed bracket")

    with pytest.raises(ConfigError, match="Error parsing YAML"):
        load_config(config_file)

def test_load_config_missing_pseudo_dir(tmp_path):
    """Verify error when pseudo dir is missing."""
    config_data = {
        "global": {
            "project_name": "TestProject",
            "database_path": str(tmp_path / "test.db")
        },
        "dft": {
            "command": "pw.x",
            "pseudopotential_dir": str(tmp_path / "non_existent_pseudos")
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ConfigError, match="Pseudopotential directory does not exist"):
        load_config(config_file)
