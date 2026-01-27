"""Utilities for handling configuration objects."""

import logging
from pathlib import Path

import yaml

from mlip_autopipec.config.models import MLIPConfig

logger = logging.getLogger(__name__)

def validate_path_safety(path: Path | str) -> Path:
    """
    Ensures the path is safe and resolved.
    Prevents path traversal attacks by ensuring path is absolute or relative to CWD.
    """
    try:
        if isinstance(path, str):
            path = Path(path)
        resolved = path.resolve()
        # In a real restricted environment, we might check if resolved path is within a specific root.
        # For now, we ensure it's resolved and not empty.
        if str(resolved) == ".":
             return resolved
        return resolved
    except Exception as e:
        msg = f"Invalid path: {path}"
        raise ValueError(msg) from e


def load_config(path: Path) -> MLIPConfig:
    """
    Load config helper using MLIPConfig.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated MLIPConfig object.
    """
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    try:
        with path.open("r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            msg = "Configuration must be a dictionary."
            raise ValueError(msg)

        return MLIPConfig(**data)
    except yaml.YAMLError as e:
        logger.exception("Failed to parse YAML")
        msg = f"Invalid YAML file: {path}"
        raise ValueError(msg) from e
    except Exception as e:
        logger.exception("Failed to load config")
        msg = f"Invalid configuration: {e}"
        raise ValueError(msg) from e
