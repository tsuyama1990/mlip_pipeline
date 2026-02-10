from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models import Config, GeneratorType


def test_load_valid_config(tmp_path: Path) -> None:
    data = {
        "orchestrator": {
            "work_dir": str(tmp_path / "work"),
            "max_iterations": 5
        },
        "generator": {
            "type": "mock",
            "params": {"foo": "bar"}
        },
        "oracle": {
            "type": "mock"
        },
        "trainer": {
            "type": "mock",
            "dataset_path": str(tmp_path / "data.xyz")
        }
    }
    config = Config.model_validate(data)
    assert config.orchestrator.max_iterations == 5
    assert config.generator.type == GeneratorType.MOCK
    assert config.generator.params["foo"] == "bar"

def test_load_invalid_config_missing_field(tmp_path: Path) -> None:
    data = {
        "orchestrator": {"work_dir": str(tmp_path)},
        # missing generator, oracle, trainer
    }
    with pytest.raises(ValidationError):
        Config.model_validate(data)

def test_load_invalid_config_wrong_discriminator(tmp_path: Path) -> None:
    data = {
        "orchestrator": {"work_dir": str(tmp_path)},
        "generator": {"type": "super_fancy_generator"}, # Invalid type
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock", "dataset_path": "foo"}
    }
    with pytest.raises(ValidationError):
        Config.model_validate(data)
