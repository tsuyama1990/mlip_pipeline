import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Safely load a YAML file.
    """
    with Path(path).open("r") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def save_json(data: dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """
    Save data to a JSON file using atomic write (write to .tmp then rename).
    """
    path = Path(path)
    temp_path = path.with_suffix(".tmp")
    try:
        with temp_path.open("w") as f:
            json.dump(data, f, indent=indent)
        temp_path.rename(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def load_json(path: str | Path) -> dict[str, Any]:
    """
    Load data from a JSON file.
    """
    with Path(path).open("r") as f:
        return json.load(f)  # type: ignore[no-any-return]
