"""I/O utilities for YAML and file handling."""

import json
from pathlib import Path
from typing import Any

import yaml

MAX_CONFIG_SIZE = 1024 * 1024 * 5  # 5 MB limit for config/state files


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file safely.

    WARNING: This loads the entire file into memory. Do not use for large datasets.

    This function uses `yaml.safe_load` to prevent arbitrary code execution,
    ensuring security against malicious YAML files. It also checks file size
    to prevent OOM denial-of-service.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary containing the YAML data.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        ValueError: If file exceeds MAX_CONFIG_SIZE.
    """
    path = Path(path)
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    # Check size
    if path.stat().st_size > MAX_CONFIG_SIZE:
        msg = f"File {path} exceeds maximum allowed size ({MAX_CONFIG_SIZE} bytes)."
        raise ValueError(msg)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        # Handle empty file or non-dict root
        return {}

    return data


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Dump a dictionary to a YAML file.

    Args:
        data: Dictionary to dump.
        path: Path to the output file.
    """
    path = Path(path)
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save data to a JSON file (used for state persistence).

    Args:
        data: Dictionary to save.
        path: Path to the output file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load data from a JSON file.

    WARNING: This loads the entire file into memory. Do not use for large datasets.

    Args:
        path: Path to the JSON file.

    Returns:
        Dictionary containing the data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file exceeds MAX_CONFIG_SIZE or is not a JSON object.
    """
    path = Path(path)
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    # Check size
    if path.stat().st_size > MAX_CONFIG_SIZE:
        msg = f"File {path} exceeds maximum allowed size ({MAX_CONFIG_SIZE} bytes)."
        raise ValueError(msg)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        msg = "JSON file must contain a JSON object (dict)"
        raise TypeError(msg)

    return data
