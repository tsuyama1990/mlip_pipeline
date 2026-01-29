"""I/O utilities for YAML and file handling."""

import json
from pathlib import Path
from typing import Any, Generator, TypeVar

import ijson
import yaml
from pydantic import BaseModel

MAX_CONFIG_SIZE = 1024 * 1024 * 5  # 5 MB limit for config/state files
T = TypeVar("T", bound=BaseModel)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file safely.

    WARNING: This loads the entire file into memory. Do not use for large datasets.

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

    Raises:
        IOError: If writing fails.
    """
    path = Path(path)
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        msg = f"Failed to dump YAML to {path}: {e}"
        raise IOError(msg) from e


def load_pydantic_from_yaml(path: str | Path, model_cls: type[T]) -> T:
    """Load a Pydantic model from a YAML file.

    Args:
        path: Path to the YAML file.
        model_cls: The Pydantic model class.

    Returns:
        Instance of the Pydantic model.
    """
    data = load_yaml(path)
    return model_cls(**data)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save data to a JSON file (used for state persistence).

    Args:
        data: Dictionary to save.
        path: Path to the output file.

    Raises:
        IOError: If writing fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        msg = f"Failed to save JSON to {path}: {e}"
        raise IOError(msg) from e


def load_json(path: str | Path) -> dict[str, Any]:
    """Load data from a JSON file.

    WARNING: This loads the entire file into memory. Do not use for large datasets.

    Args:
        path: Path to the JSON file.

    Returns:
        Dictionary containing the data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file exceeds MAX_CONFIG_SIZE.
        TypeError: If file does not contain a dictionary.
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
        msg = f"JSON file {path} must contain a dictionary, got {type(data)}"
        raise TypeError(msg)

    return data


def load_json_iter(path: str | Path, item_prefix: str = "item") -> Generator[Any, None, None]:
    """Load data from a JSON file iteratively using ijson.

    This avoids loading the entire file into memory.

    Args:
        path: Path to the JSON file.
        item_prefix: The prefix path to iterate over (e.g., 'item' for a list).

    Yields:
        Parsed items from the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("rb") as f:
        yield from ijson.items(f, item_prefix)
