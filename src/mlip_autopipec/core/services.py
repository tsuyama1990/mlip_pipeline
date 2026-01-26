from pathlib import Path
from mlip_autopipec.config.models import MLIPConfig
from mlip_autopipec.utils.config_utils import load_config as _load_config_impl

def load_config(path: Path) -> MLIPConfig:
    """
    Loads and validates the configuration from a YAML file.
    Delegates to utils.config_utils.load_config.
    """
    return _load_config_impl(path)
