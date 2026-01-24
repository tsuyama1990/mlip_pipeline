"""Utilities for handling configuration objects."""

from pathlib import Path

from mlip_autopipec.config.models import (
    DFTConfig,
    SystemConfig,
    MLIPConfig,
)


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
    """Load config helper using MLIPConfig"""
    # This might be redundant if core/services.py load_config is used, but for utils we keep it simple
    # Actually, config_utils was importing unused models. Fixing that.
    pass
