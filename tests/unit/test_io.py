import json
from pathlib import Path

from mlip_autopipec.utils.io import load_json, load_yaml, save_json


def test_load_yaml_success(tmp_path: Path) -> None:
    f = tmp_path / "test.yaml"
    f.write_text("key: value")
    data = load_yaml(f)
    assert data["key"] == "value"


def test_save_json_atomic(tmp_path: Path) -> None:
    f = tmp_path / "test.json"
    data = {"key": "value"}
    save_json(data, f)
    assert f.exists()

    with f.open("r") as fh:
        loaded = json.load(fh)
    assert loaded["key"] == "value"


def test_load_json_success(tmp_path: Path) -> None:
    f = tmp_path / "test.json"
    f.write_text('{"key": "value"}')
    data = load_json(f)
    assert data["key"] == "value"
