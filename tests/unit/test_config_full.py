"""Unit tests for full configuration."""

from pathlib import Path

import pytest
import yaml

from pyacemaker.core.config import load_config
from pyacemaker.core.exceptions import ConfigurationError


def test_full_config_loading(tmp_path: Path) -> None:
    """Test loading a complete configuration file."""
    pseudo_path = tmp_path / "Fe.pbe.UPF"
    pseudo_path.touch()

    config_dict = {
        "version": "0.1.0",
        "project": {"name": "TestProject", "root_dir": str(tmp_path)},
        "oracle": {
            "dft": {"code": "quantum_espresso", "pseudopotentials": {"Fe": str(pseudo_path)}}
        },
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
    pseudo_path = tmp_path / "Fe.pbe.UPF"
    pseudo_path.touch()

    config_dict = {
        "project": {"name": "Minimal", "root_dir": str(tmp_path)},
        "oracle": {"dft": {"code": "vasp", "pseudopotentials": {"Fe": str(pseudo_path)}}},
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
        "oracle": {},  # Missing dft
    }

    config_file = tmp_path / "invalid.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_dict, f)

    # load_config raises ConfigurationError wrapping ValidationError

    with pytest.raises(ConfigurationError):
        load_config(config_file)


def test_dynamics_config(tmp_path: Path) -> None:
    """Test DynamicsEngineConfig fields."""
    pseudo_path = tmp_path / "Fe.pbe.UPF"
    pseudo_path.touch()

    config_dict = {
        "version": "0.1.0",
        "project": {"name": "TestProject", "root_dir": str(tmp_path)},
        "oracle": {
            "dft": {
                "code": "quantum_espresso",
                "pseudopotentials": {"Fe": str(pseudo_path)},
            }
        },
        "dynamics_engine": {
            "engine": "lammps",
            "gamma_threshold": 5.0,
            "timestep": 0.002,
            "temperature": 500.0,
            "pressure": 1.0,
            "n_steps": 50000,
            "hybrid_baseline": "lj",
        },
    }

    config_file = tmp_path / "dynamics_config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_dict, f)

    config = load_config(config_file)
    dyn = config.dynamics_engine
    assert dyn.engine == "lammps"
    assert dyn.gamma_threshold == 5.0
    assert dyn.timestep == 0.002
    assert dyn.temperature == 500.0
    assert dyn.pressure == 1.0
    assert dyn.n_steps == 50000
    assert dyn.hybrid_baseline == "lj"


def test_dynamics_config_defaults(tmp_path: Path) -> None:
    """Test DynamicsEngineConfig defaults."""
    pseudo_path = tmp_path / "Fe.pbe.UPF"
    pseudo_path.touch()

    config_dict = {
        "version": "0.1.0",
        "project": {"name": "TestProject", "root_dir": str(tmp_path)},
        "oracle": {
            "dft": {
                "code": "quantum_espresso",
                "pseudopotentials": {"Fe": str(pseudo_path)},
            }
        },
    }

    config_file = tmp_path / "dynamics_defaults.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_dict, f)

    config = load_config(config_file)
    dyn = config.dynamics_engine
    assert dyn.timestep == 0.001
    assert dyn.hybrid_baseline == "zbl"


def test_dynamics_config_validation(tmp_path: Path) -> None:
    """Test DynamicsEngineConfig validation."""
    pseudo_path = tmp_path / "Fe.pbe.UPF"
    pseudo_path.touch()

    config_dict = {
        "version": "0.1.0",
        "project": {"name": "TestProject", "root_dir": str(tmp_path)},
        "oracle": {
            "dft": {
                "code": "quantum_espresso",
                "pseudopotentials": {"Fe": str(pseudo_path)},
            }
        },
        "dynamics_engine": {
            "hybrid_baseline": "invalid_baseline",
        },
    }

    config_file = tmp_path / "dynamics_invalid.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_dict, f)

    with pytest.raises(ConfigurationError, match="Invalid hybrid_baseline"):
        load_config(config_file)
