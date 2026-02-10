from pathlib import Path

import pytest
import yaml

from mlip_autopipec.config import load_config
from mlip_autopipec.domain_models import Config


def test_load_config_success(tmp_path: Path) -> None:
    config_data = {
        "orchestrator": {"work_dir": str(tmp_path), "max_iterations": 1},
        "generator": {"type": "mock"},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock", "dataset_path": str(tmp_path / "data")}
    }
    p = tmp_path / "config.yaml"
    with p.open("w") as f:
        yaml.dump(config_data, f)

    config = load_config(p)
    assert isinstance(config, Config)

def test_load_config_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "non_existent.yaml")
