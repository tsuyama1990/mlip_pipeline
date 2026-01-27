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
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open() as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        msg = "Configuration file must contain a dictionary."
        raise TypeError(msg)

    return MLIPConfig.model_validate(data)
