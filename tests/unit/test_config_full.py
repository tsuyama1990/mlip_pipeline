"""Unit tests for full configuration."""

from pathlib import Path

import pytest
import yaml

from pyacemaker.core.config import load_config


def test_full_config_loading(tmp_path: Path) -> None:
    """Test loading a complete configuration file."""
    config_dict = {
        "version": "0.1.0",
        "project": {"name": "TestProject", "root_dir": str(tmp_path)},
        "oracle": {"dft": {"code": "quantum_espresso"}},
        "structure_generator": {"strategy": "adaptive"},
        "trainer": {"potential_type": "pace"},
        "dynamics_engine": {"engine": "lammps"},
        "validator": {"metrics": ["rmse_energy"]},
        "orchestrator": {"max_cycles": 5},
    }

    config_file = tmp_path / "full_config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_dict, f)

    config = load_config(config_file)
    assert config.project.name == "TestProject"
    assert config.structure_generator.strategy == "adaptive"
    assert config.oracle.dft.code == "quantum_espresso"
    assert config.orchestrator.max_cycles == 5


def test_minimal_config_defaults(tmp_path: Path) -> None:
    """Test loading a minimal configuration using defaults."""
    config_dict = {
        "project": {"name": "Minimal", "root_dir": str(tmp_path)},
        "oracle": {"dft": {"code": "vasp"}}
    }

    config_file = tmp_path / "minimal.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_dict, f)

    config = load_config(config_file)
    assert config.structure_generator.strategy == "random"  # Default
    assert config.orchestrator.max_cycles == 10  # Default


def test_invalid_config_structure(tmp_path: Path) -> None:
    """Test invalid configuration structure."""
    config_dict = {
        "project": {"name": "Invalid"},
        "oracle": {}  # Missing dft
    }

    config_file = tmp_path / "invalid.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_dict, f)

    # load_config raises ConfigurationError wrapping ValidationError
    from pyacemaker.core.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError):
        load_config(config_file)
