
import pytest
import yaml

from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.config.schemas.system import SystemConfig


@pytest.fixture
def valid_yaml(tmp_path):
    data = {
        "project_name": "FactoryTest",
        "target_system": {
            "elements": ["Al", "Cu"],
            "composition": {"Al": 0.9, "Cu": 0.1},
            "crystal_structure": "fcc"
        },
        "simulation_goal": {
            "type": "melt_quench"
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 8
        }
    }
    p = tmp_path / "input.yaml"
    with p.open("w") as f:
        yaml.dump(data, f)
    return p

def test_factory_from_yaml(valid_yaml, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Pass a Path object for creating workspace inside tmp_path to avoid creating real dirs
    # But ConfigFactory creates dirs based on project_name or config.
    # We should probably mock creating dirs or ensure it creates in tmp_path.
    # The factory logic usually resolves paths relative to input file or CWD.

    # Let's assume ConfigFactory.from_yaml returns a SystemConfig
    config = ConfigFactory.from_yaml(valid_yaml)

    assert isinstance(config, SystemConfig)
    assert config.project_name == "FactoryTest"
    assert config.target_system.elements == ["Al", "Cu"]
    assert config.resources.parallel_cores == 8

    # Check absolute paths
    assert config.working_dir.is_absolute()
    assert config.db_path.endswith(".db")

def test_factory_invalid_yaml(tmp_path):
    from pydantic import ValidationError
    p = tmp_path / "bad.yaml"
    p.write_text("project_name: 'NoResources'")

    with pytest.raises(ValidationError):
        ConfigFactory.from_yaml(p)
