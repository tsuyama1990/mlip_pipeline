from pathlib import Path

import yaml

from mlip_autopipec.config.models import MLIPConfig


def load_config(path: Path) -> MLIPConfig:
    """
    Loads and validates the configuration from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Validated MLIPConfig object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValidationError: If the configuration is invalid.
        yaml.YAMLError: If the file is not valid YAML.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a dictionary.")

    return MLIPConfig.model_validate(data)
