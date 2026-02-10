from pathlib import Path

import yaml

from mlip_autopipec.domain_models import Config


def load_config(path: Path) -> Config:
    """
    Loads and validates the configuration from a YAML file.
    """
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("r") as f:
        data = yaml.safe_load(f)

    return Config.model_validate(data)
