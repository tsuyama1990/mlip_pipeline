from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mlip_autopipec.core.config_parser import _substitute_env, load_config


def test_substitute_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_VAR", "value")
    assert _substitute_env("${TEST_VAR}") == "value"
    assert _substitute_env("$TEST_VAR") == "value"
    assert _substitute_env("prefix_${TEST_VAR}_suffix") == "prefix_value_suffix"
    assert _substitute_env("path/$TEST_VAR/file") == "path/value/file"


def test_substitute_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_VAR", raising=False)
    with pytest.raises(ValueError, match="Environment variable 'MISSING_VAR' not set"):
        _substitute_env("${MISSING_VAR}")


def test_load_config_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME_DIR", str(tmp_path))
    config_data = {
        "orchestrator": {"work_dir": "${HOME_DIR}/work"},
        "generator": {"type": "RANDOM", "num_structures": 5},
        "oracle": {"type": "QUANTUM_ESPRESSO", "command": "pw.x"},
        "trainer": {"type": "PACEMAKER"},
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)
    assert config.orchestrator.work_dir == tmp_path / "work"
    assert config.generator.num_structures == 5


def test_load_config_invalid(tmp_path: Path) -> None:
    config_data = {
        "orchestrator": {"work_dir": "wd"},
        # Missing generator
        "oracle": {"type": "QUANTUM_ESPRESSO", "command": "cmd"},
        "trainer": {"type": "PACEMAKER"},
    }
    config_file = tmp_path / "bad_config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ValidationError):
        load_config(config_file)


def test_walk_list(monkeypatch: pytest.MonkeyPatch) -> None:
    from mlip_autopipec.core.config_parser import _walk_and_substitute

    monkeypatch.setenv("VAR", "val")
    data = ["${VAR}", "static"]
    assert _walk_and_substitute(data) == ["val", "static"]
