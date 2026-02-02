from pathlib import Path

import yaml

from mlip_autopipec.config import Config


def load_config(path: Path) -> Config:
    """
    Loads configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Config: Validated configuration object.

    Raises:
        FileNotFoundError: If file not found.
        yaml.YAMLError: If invalid YAML.
        ValidationError: If invalid schema.
    """
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("r") as f:
        data = yaml.safe_load(f)

    return Config.model_validate(data)
