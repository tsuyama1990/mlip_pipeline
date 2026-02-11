from pathlib import Path
from typing import Any

import pytest

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models import GeneratorType, GlobalConfig


def test_orchestrator_initialization(valid_config_dict: dict[str, Any], tmp_path: Path) -> None:
    # Update work_dir to tmp_path
    valid_config_dict["orchestrator"]["work_dir"] = tmp_path
    config = GlobalConfig(**valid_config_dict)

    orchestrator = Orchestrator(config)
    assert orchestrator.config == config
    # Check if work_dir is created (it should be)
    assert tmp_path.exists()

def test_orchestrator_run(valid_config_dict: dict[str, Any], tmp_path: Path) -> None:
    valid_config_dict["orchestrator"]["work_dir"] = tmp_path
    config = GlobalConfig(**valid_config_dict)
    orchestrator = Orchestrator(config)

    # Run mock workflow
    orchestrator.run()

def test_orchestrator_invalid_component_type(valid_config_dict: dict[str, Any]) -> None:
    valid_config_dict["generator"]["type"] = GeneratorType.RANDOM
    config = GlobalConfig(**valid_config_dict)

    with pytest.raises(NotImplementedError):
        Orchestrator(config)
