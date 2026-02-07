from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from mlip_autopipec.domain_models import GlobalConfig, Structure


def test_scenario_1_1_config_parsing(tmp_path: Path) -> None:
    # 1. Create a config.yaml
    config_data = {
        "workdir": str(tmp_path / "workdir"),
        "max_cycles": 10,
        "oracle": {"type": "mock", "params": {}},
        "trainer": {"type": "mock", "params": {}},
        "dynamics": {"type": "mock", "params": {}},
        "generator": {"type": "mock", "params": {}},
        "validator": {"type": "mock", "params": {}},
        "selector": {"type": "mock", "params": {}},
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    # 2. Load it
    with config_file.open() as f:
        loaded_data = yaml.safe_load(f)
    config = GlobalConfig(**loaded_data)

    assert config.oracle.type == "mock"
    assert config.workdir == tmp_path / "workdir"


def test_scenario_1_1_structure_validation() -> None:
    # 3. Create Structure with valid/invalid arrays
    # Valid
    s = Structure(
        symbols=["H"],
        positions=[[0.0, 0.0, 0.0]],  # type: ignore[arg-type]
        cell=np.eye(3).tolist(),  # type: ignore[arg-type]
        pbc=[True, True, True],  # type: ignore[arg-type]
    )
    assert s.positions.shape == (1, 3)

    # Invalid
    with pytest.raises(ValueError, match="Positions must be"):
        Structure(
            symbols=["H"],
            positions=[[0.0, 0.0]],  # type: ignore[arg-type]
            cell=np.eye(3).tolist(),  # type: ignore[arg-type]
            pbc=[True, True, True],  # type: ignore[arg-type]
        )
