from pathlib import Path

import yaml

from mlip_autopipec.constants import MAX_CONFIG_SIZE_BYTES
from mlip_autopipec.domain_models import GlobalConfig


def load_config(config_path: Path) -> GlobalConfig:
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    # Size check
    if config_path.stat().st_size > MAX_CONFIG_SIZE_BYTES:
        msg = f"Config file too large (> {MAX_CONFIG_SIZE_BYTES} bytes)"
        raise ValueError(msg)

    with config_path.open('r') as f:
        data = yaml.safe_load(f)

    return GlobalConfig.model_validate(data)
