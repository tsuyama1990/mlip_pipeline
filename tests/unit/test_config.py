from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import (
    GlobalConfig,
    MockOracleConfig,
    MockTrainerConfig,
    QuantumEspressoConfig,
)


def test_global_config_mock_defaults(tmp_path: Path) -> None:
    config_data = {
        "workdir": str(tmp_path / "work"),
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    config = GlobalConfig.from_yaml(config_file)

    assert config.workdir == tmp_path / "work"
    assert isinstance(config.oracle, MockOracleConfig)
    assert isinstance(config.trainer, MockTrainerConfig)
    assert config.max_cycles == 10


def test_global_config_polymorphism(tmp_path: Path) -> None:
    config_data = {
        "workdir": str(tmp_path / "work"),
        "oracle": {
            "type": "qe",
            "command": "pw.x",
            "pseudo_dir": "/path/to/pseudos",
            "pseudopotentials": {"H": "H.upf"},
        },
        "trainer": {"type": "mock"},
    }
    config_file = tmp_path / "config_poly.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    config = GlobalConfig.from_yaml(config_file)

    assert isinstance(config.oracle, QuantumEspressoConfig)
    assert config.oracle.command == "pw.x"
    assert config.oracle.pseudopotentials["H"] == "H.upf"
    assert isinstance(config.trainer, MockTrainerConfig)


def test_global_config_invalid_type(tmp_path: Path) -> None:
    config_data = {
        "workdir": str(tmp_path / "work"),
        "oracle": {
            "type": "unknown",
        },
    }
    config_file = tmp_path / "config_invalid.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ValidationError):
        GlobalConfig.from_yaml(config_file)


def test_global_config_missing_required(tmp_path: Path) -> None:
    config_data = {
        # missing workdir
        "max_cycles": 5
    }
    config_file = tmp_path / "config_missing.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ValidationError):
        GlobalConfig.from_yaml(config_file)
