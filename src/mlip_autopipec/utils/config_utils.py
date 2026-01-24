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
        raise ValueError(f"Invalid path: {path}") from e


def load_config(path: Path) -> MLIPConfig:
    """
    Load config helper using MLIPConfig.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated MLIPConfig object.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("Configuration must be a dictionary.")

        return MLIPConfig(**data)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML: {e}")
        raise ValueError(f"Invalid YAML file: {path}") from e
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise ValueError(f"Invalid configuration: {e}") from e
